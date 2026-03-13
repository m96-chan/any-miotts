use std::any::Any;
use std::fmt;
use std::path::Path;
use std::time::Duration;

use crate::device::DeviceInfo;
use crate::error::TtsError;

/// Which model component to load/run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelComponent {
    /// WavLM + GlobalEncoder: reference wav → 128-dim speaker embedding
    SpeakerEncoder,
    /// LFM2: autoregressive codec token generation
    Lfm2,
    /// MioCodec: codec tokens + speaker embedding → PCM waveform
    MioCodec,
}

impl fmt::Display for ModelComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpeakerEncoder => write!(f, "SpeakerEncoder"),
            Self::Lfm2 => write!(f, "LFM2"),
            Self::MioCodec => write!(f, "MioCodec"),
        }
    }
}

/// Tensor data exchanged between backends.
///
/// `F32` and `U32` variants enable cross-backend transfer.
/// `Native` allows zero-copy within the same backend.
pub enum TensorData {
    F32 {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    U32 {
        data: Vec<u32>,
        shape: Vec<usize>,
    },
    /// Backend-native tensor (zero-copy within same backend).
    Native(Box<dyn Any + Send + Sync>),
}

/// Result of a benchmark run for a specific component.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Which component was benchmarked.
    pub component: ModelComponent,
    /// Average forward pass duration.
    pub avg_duration: Duration,
    /// Number of measurement iterations (excluding warmup).
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Throughput as operations per second.
    pub fn ops_per_sec(&self) -> f64 {
        1.0 / self.avg_duration.as_secs_f64()
    }
}

/// Opaque handle to a loaded model within a backend.
pub trait LoadedModel: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Opaque LFM2 generation state (KV caches, conv states).
pub trait Lfm2State: Send {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Current sequence offset (number of tokens processed so far).
    fn seq_offset(&self) -> usize;
}

/// Backend trait: the core abstraction for multi-backend inference.
///
/// Each backend implements the subset of methods for the components it supports.
/// Unsupported methods return `Err(TtsError::Unsupported(...))` by default.
pub trait Backend: Send + Sync {
    /// Human-readable backend name (e.g. "candle-cuda", "llama.cpp-opencl").
    fn name(&self) -> &str;

    /// Which model components this backend can handle.
    fn supported_components(&self) -> &[ModelComponent];

    /// Device this backend runs on.
    fn device_info(&self) -> &DeviceInfo;

    /// Download and/or locate model files for the given component.
    /// Returns a path to the model directory.
    fn ensure_model(&self, component: ModelComponent) -> Result<(), TtsError> {
        Err(TtsError::Unsupported(format!(
            "{}: ensure_model not implemented for {component}",
            self.name()
        )))
    }

    /// Load a model component from disk. Returns an opaque handle.
    fn load_model(
        &self,
        component: ModelComponent,
        model_dir: &Path,
    ) -> Result<Box<dyn LoadedModel>, TtsError> {
        let _ = (component, model_dir);
        Err(TtsError::Unsupported(format!(
            "{}: load_model not implemented",
            self.name()
        )))
    }

    // ── SpeakerEncoder ─────────────────────────────────────────────

    /// Encode a reference waveform (16kHz mono f32 PCM) into a speaker embedding.
    /// Returns TensorData::F32 with shape [128].
    fn encode_speaker(
        &self,
        model: &dyn LoadedModel,
        wav_16k_mono: &[f32],
    ) -> Result<TensorData, TtsError> {
        let _ = (model, wav_16k_mono);
        Err(TtsError::Unsupported(format!(
            "{}: encode_speaker not supported",
            self.name()
        )))
    }

    // ── LFM2 ───────────────────────────────────────────────────────

    /// Run LFM2 prefill on the given token IDs. Returns opaque generation state.
    /// Also returns logits for the last position as Vec<f32>.
    fn lfm2_prefill(
        &self,
        model: &dyn LoadedModel,
        input_ids: &[u32],
    ) -> Result<(Box<dyn Lfm2State>, Vec<f32>), TtsError> {
        let _ = (model, input_ids);
        Err(TtsError::Unsupported(format!(
            "{}: lfm2_prefill not supported",
            self.name()
        )))
    }

    /// Run one LFM2 decode step. Returns logits for the next position as Vec<f32>.
    fn lfm2_decode_step(
        &self,
        model: &dyn LoadedModel,
        state: &mut dyn Lfm2State,
        token: u32,
    ) -> Result<Vec<f32>, TtsError> {
        let _ = (model, state, token);
        Err(TtsError::Unsupported(format!(
            "{}: lfm2_decode_step not supported",
            self.name()
        )))
    }

    // ── MioCodec ───────────────────────────────────────────────────

    /// Decode codec tokens + speaker embedding into PCM f32 waveform.
    fn miocodec_decode(
        &self,
        model: &dyn LoadedModel,
        tokens: &[u32],
        speaker_embedding: &TensorData,
    ) -> Result<Vec<f32>, TtsError> {
        let _ = (model, tokens, speaker_embedding);
        Err(TtsError::Unsupported(format!(
            "{}: miocodec_decode not supported",
            self.name()
        )))
    }

    // ── Benchmark ──────────────────────────────────────────────────

    /// Run a small benchmark for the given component.
    /// Default: 3 warmup + 5 measured iterations.
    fn benchmark(
        &self,
        model: &dyn LoadedModel,
        component: ModelComponent,
    ) -> Result<BenchmarkResult, TtsError> {
        let _ = (model, component);
        Err(TtsError::Unsupported(format!(
            "{}: benchmark not supported",
            self.name()
        )))
    }
}
