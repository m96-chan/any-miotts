//! [`Backend`] implementation for llama.cpp, targeting LFM2 inference only.
//!
//! This backend loads a GGUF-quantised LFM2 model via llama.cpp and exposes
//! prefill / decode-step operations that return raw logits, leaving sampling
//! to the core scheduler.

use std::any::Any;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use tracing::info;

use any_miotts_core::backend::{
    Backend, BenchmarkResult, Lfm2State, LoadedModel, ModelComponent, TensorData,
};
use any_miotts_core::device::DeviceInfo;
use any_miotts_core::error::TtsError;

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for the llama.cpp backend.
#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    /// Number of layers to offload to GPU.  Set to `u32::MAX` for all layers.
    /// Set to `0` for CPU-only inference.
    pub n_gpu_layers: u32,
    /// Context window size in tokens.
    pub n_ctx: u32,
    /// Maximum batch size (number of tokens per decode call).
    pub n_batch: u32,
    /// Number of threads for CPU compute.
    pub n_threads: u32,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 99, // offload everything if GPU available
            n_ctx: 4096,
            n_batch: 512,
            n_threads: 4,
        }
    }
}

// ── Loaded model wrapper ──────────────────────────────────────────────

/// Holds the llama.cpp model and its lazily-initialised backend handle.
///
/// llama.cpp requires the `LlamaBackend` to stay alive for the lifetime of
/// the model, so we bundle them together.
pub struct LoadedLlamaCppLfm2 {
    #[allow(dead_code)]
    backend: LlamaBackend,
    model: LlamaModel,
    config: LlamaCppConfig,
    vocab_size: i32,
}

// SAFETY: LlamaModel and LlamaBackend are thread-safe in practice (the C
// library serialises access internally).  The Rust wrapper marks LlamaModel
// as Send+Sync.
unsafe impl Send for LoadedLlamaCppLfm2 {}
unsafe impl Sync for LoadedLlamaCppLfm2 {}

impl LoadedModel for LoadedLlamaCppLfm2 {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── LFM2 generation state ─────────────────────────────────────────────

/// Opaque generation state wrapping a llama.cpp context with its KV cache.
///
/// The context is behind a `Mutex` because `LlamaContext` is `!Sync` (it
/// holds mutable C state), but `Lfm2State` requires `Send`.
pub struct LlamaCppLfm2State {
    /// The llama.cpp context holding the KV cache.
    /// Wrapped in Mutex because LlamaContext is !Sync but Lfm2State: Send.
    ctx: Mutex<LlamaCppContextInner>,
    /// Number of tokens processed so far (prompt + generated).
    offset: usize,
}

/// Inner context holder (not Send/Sync by default).
struct LlamaCppContextInner {
    ctx: llama_cpp_2::context::LlamaContext<'static>,
}

// SAFETY: We ensure single-threaded access through the Mutex.
unsafe impl Send for LlamaCppContextInner {}
unsafe impl Sync for LlamaCppContextInner {}

impl Lfm2State for LlamaCppLfm2State {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn seq_offset(&self) -> usize {
        self.offset
    }
}

// ── Backend ───────────────────────────────────────────────────────────

/// llama.cpp backend for LFM2 autoregressive inference.
///
/// This backend supports **only** [`ModelComponent::Lfm2`].  Speaker encoding
/// and MioCodec decoding must be handled by another backend (e.g. candle).
pub struct LlamaCppBackend {
    device_info: DeviceInfo,
    config: LlamaCppConfig,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend with the given device info and config.
    pub fn new(device_info: DeviceInfo, config: LlamaCppConfig) -> Self {
        Self {
            device_info,
            config,
        }
    }

    /// Create a backend using default config and auto-discovered device.
    pub fn with_defaults() -> Self {
        let device_info = super::discovery::discover_device();
        Self {
            device_info,
            config: LlamaCppConfig::default(),
        }
    }
}

impl Backend for LlamaCppBackend {
    fn name(&self) -> &str {
        if self.device_info.kind.is_gpu() {
            "llama.cpp-opencl"
        } else {
            "llama.cpp-cpu"
        }
    }

    fn supported_components(&self) -> &[ModelComponent] {
        &[ModelComponent::Lfm2]
    }

    fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    fn load_model(
        &self,
        component: ModelComponent,
        model_dir: &Path,
    ) -> Result<Box<dyn LoadedModel>, TtsError> {
        if component != ModelComponent::Lfm2 {
            return Err(TtsError::Unsupported(format!(
                "{}: only Lfm2 is supported, got {component}",
                self.name()
            )));
        }

        // Locate the GGUF file.  Convention: model_dir points to either
        // a directory containing *.gguf or directly to a .gguf file.
        let gguf_path = if model_dir.is_file() {
            model_dir.to_path_buf()
        } else {
            find_gguf_in_dir(model_dir)?
        };

        info!("Initialising llama.cpp backend...");
        let backend = LlamaBackend::init().map_err(|e| {
            TtsError::Model(format!("Failed to initialise llama.cpp backend: {e}"))
        })?;

        let model_params = {
            let params = LlamaModelParams::default()
                .with_n_gpu_layers(self.config.n_gpu_layers);
            // LlamaModelParams requires pinning for safety.
            Box::pin(params)
        };

        info!("Loading GGUF model from {}", gguf_path.display());
        let model =
            LlamaModel::load_from_file(&backend, &gguf_path, &model_params).map_err(|e| {
                TtsError::Model(format!("Failed to load GGUF model: {e}"))
            })?;

        let vocab_size = model.n_vocab();
        info!(
            "LFM2 GGUF loaded: {} layers, {} params, vocab {}",
            model.n_layer(),
            model.n_params(),
            vocab_size,
        );

        Ok(Box::new(LoadedLlamaCppLfm2 {
            backend,
            model,
            config: self.config.clone(),
            vocab_size,
        }))
    }

    // ── LFM2 prefill ──────────────────────────────────────────────────

    fn lfm2_prefill(
        &self,
        model: &dyn LoadedModel,
        input_ids: &[u32],
    ) -> Result<(Box<dyn Lfm2State>, Vec<f32>), TtsError> {
        let loaded = model
            .as_any()
            .downcast_ref::<LoadedLlamaCppLfm2>()
            .ok_or_else(|| TtsError::Model("Expected LoadedLlamaCppLfm2".into()))?;

        // Create a fresh context for this generation session.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(loaded.config.n_ctx))
            .with_n_batch(loaded.config.n_batch)
            .with_n_threads(loaded.config.n_threads as i32)
            .with_n_threads_batch(loaded.config.n_threads as i32);

        // SAFETY: We keep the model alive via the LoadedLlamaCppLfm2 handle
        // which is held by the caller.  We transmute the lifetime to 'static
        // so it can be stored in the state struct; the caller must ensure the
        // model outlives the state (which the TtsEngine guarantees).
        let ctx = unsafe {
            let model_ref: &LlamaModel = &loaded.model;
            let model_static: &'static LlamaModel =
                &*(model_ref as *const LlamaModel);
            model_static
                .new_context(&loaded.backend, ctx_params)
                .map_err(|e| TtsError::Inference(format!("Create llama context: {e}")))?
        };

        let mut ctx = ctx;
        let seq_len = input_ids.len();

        // Build a batch with all prompt tokens.
        let mut batch = LlamaBatch::new(seq_len, 1);

        for (pos, &tok) in input_ids.iter().enumerate() {
            let is_last = pos == seq_len - 1;
            batch
                .add(LlamaToken(tok as i32), pos as i32, &[0], is_last)
                .map_err(|e| TtsError::Inference(format!("Batch add: {e}")))?;
        }

        // Decode the full prompt.
        ctx.decode(&mut batch)
            .map_err(|e| TtsError::Inference(format!("Prefill decode: {e}")))?;

        // Extract logits for the last token.
        let logits = extract_logits(&ctx, (seq_len - 1) as i32, loaded.vocab_size)?;

        let state = LlamaCppLfm2State {
            ctx: Mutex::new(LlamaCppContextInner { ctx }),
            offset: seq_len,
        };

        Ok((Box::new(state), logits))
    }

    // ── LFM2 decode step ──────────────────────────────────────────────

    fn lfm2_decode_step(
        &self,
        model: &dyn LoadedModel,
        state: &mut dyn Lfm2State,
        token: u32,
    ) -> Result<Vec<f32>, TtsError> {
        let loaded = model
            .as_any()
            .downcast_ref::<LoadedLlamaCppLfm2>()
            .ok_or_else(|| TtsError::Model("Expected LoadedLlamaCppLfm2".into()))?;

        let llama_state = state
            .as_any_mut()
            .downcast_mut::<LlamaCppLfm2State>()
            .ok_or_else(|| TtsError::Model("Expected LlamaCppLfm2State".into()))?;

        let mut guard = llama_state
            .ctx
            .lock()
            .map_err(|e| TtsError::Inference(format!("Lock context: {e}")))?;

        let pos = llama_state.offset as i32;

        // Single-token batch.
        let mut batch = LlamaBatch::new(1, 1);
        batch
            .add(LlamaToken(token as i32), pos, &[0], true)
            .map_err(|e| TtsError::Inference(format!("Batch add: {e}")))?;

        guard
            .ctx
            .decode(&mut batch)
            .map_err(|e| TtsError::Inference(format!("Decode step: {e}")))?;

        let logits = extract_logits(&guard.ctx, 0, loaded.vocab_size)?;

        llama_state.offset += 1;

        Ok(logits)
    }

    // ── Unsupported operations ────────────────────────────────────────

    fn encode_speaker(
        &self,
        _model: &dyn LoadedModel,
        _wav_16k_mono: &[f32],
    ) -> Result<TensorData, TtsError> {
        Err(TtsError::Unsupported(format!(
            "{}: encode_speaker not supported (LFM2 only backend)",
            self.name()
        )))
    }

    fn miocodec_decode(
        &self,
        _model: &dyn LoadedModel,
        _tokens: &[u32],
        _speaker_embedding: &TensorData,
    ) -> Result<Vec<f32>, TtsError> {
        Err(TtsError::Unsupported(format!(
            "{}: miocodec_decode not supported (LFM2 only backend)",
            self.name()
        )))
    }

    // ── Benchmark ─────────────────────────────────────────────────────

    fn benchmark(
        &self,
        model: &dyn LoadedModel,
        component: ModelComponent,
    ) -> Result<BenchmarkResult, TtsError> {
        if component != ModelComponent::Lfm2 {
            return Err(TtsError::Unsupported(format!(
                "{}: benchmark only supported for Lfm2",
                self.name()
            )));
        }

        const WARMUP: usize = 2;
        const MEASURED: usize = 5;
        let dummy_ids: Vec<u32> = vec![1; 4];

        // Warmup
        for _ in 0..WARMUP {
            let (_, _logits) = self.lfm2_prefill(model, &dummy_ids)?;
        }

        // Measured
        let mut total = Duration::ZERO;
        for _ in 0..MEASURED {
            let start = Instant::now();
            let (_, _logits) = self.lfm2_prefill(model, &dummy_ids)?;
            total += start.elapsed();
        }

        Ok(BenchmarkResult {
            component,
            avg_duration: total / MEASURED as u32,
            iterations: MEASURED,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Extract logits from the context at the given batch index.
fn extract_logits(
    ctx: &llama_cpp_2::context::LlamaContext<'_>,
    batch_idx: i32,
    vocab_size: i32,
) -> Result<Vec<f32>, TtsError> {
    let raw = ctx.get_logits_ith(batch_idx);
    if raw.len() < vocab_size as usize {
        return Err(TtsError::Inference(format!(
            "Logits slice too short: got {}, expected {}",
            raw.len(),
            vocab_size
        )));
    }
    Ok(raw[..vocab_size as usize].to_vec())
}

/// Find the first `.gguf` file in a directory.
fn find_gguf_in_dir(dir: &Path) -> Result<std::path::PathBuf, TtsError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| TtsError::Model(format!("Read dir {}: {e}", dir.display())))?;

    for entry in entries {
        let entry = entry.map_err(|e| TtsError::Model(format!("Dir entry: {e}")))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "gguf") {
            return Ok(path);
        }
    }

    Err(TtsError::Model(format!(
        "No .gguf file found in {}",
        dir.display()
    )))
}
