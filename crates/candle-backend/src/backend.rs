use std::any::Any;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tracing::info;

use candle_miotts::models::config::{Lfm2Config, MioCodecConfig};
use candle_miotts::models::download::{ensure_models_downloaded, ModelPaths};
use candle_miotts::models::lfm2::Lfm2ForCausalLM;
use candle_miotts::models::miocodec::MioCodecDecoder;
use candle_miotts::models::wavlm::{GlobalEncoder, WavLM};

use any_miotts_core::backend::{
    Backend, BenchmarkResult, Lfm2State, LoadedModel, ModelComponent, TensorData,
};
use any_miotts_core::device::DeviceInfo;
use any_miotts_core::error::TtsError;

/// Re-export ModelPaths for consumers.
pub type CandleModelPaths = ModelPaths;

/// Candle-based backend supporting CUDA, Metal, and CPU.
pub struct CandleBackend {
    device_info: DeviceInfo,
    device: Device,
    /// Cached model paths (downloaded once).
    model_paths: Option<ModelPaths>,
}

impl CandleBackend {
    pub fn new(device_info: DeviceInfo, device: Device) -> Self {
        Self {
            device_info,
            device,
            model_paths: None,
        }
    }

    /// Download models and cache paths. Must be called before load_model.
    pub async fn ensure_models(&mut self) -> Result<(), TtsError> {
        if self.model_paths.is_none() {
            let paths = ensure_models_downloaded()
                .await
                .map_err(|e| TtsError::Download(format!("{e}")))?;
            self.model_paths = Some(paths);
        }
        Ok(())
    }

    /// Get the cached model paths (Result version for internal use).
    fn paths(&self) -> Result<&ModelPaths, TtsError> {
        self.model_paths
            .as_ref()
            .ok_or_else(|| TtsError::Model("Models not downloaded yet. Call ensure_models() first.".into()))
    }

    /// Get the cached model paths (Option version for external access).
    pub fn paths_ref(&self) -> Option<&ModelPaths> {
        self.model_paths.as_ref()
    }

    /// Access the candle Device.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ── Loaded model wrappers ──────────────────────────────────────────

pub struct LoadedLfm2 {
    pub model: Lfm2ForCausalLM,
    pub config: Lfm2Config,
}

impl LoadedModel for LoadedLfm2 {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

pub struct LoadedMioCodec {
    pub model: MioCodecDecoder,
    pub config: MioCodecConfig,
}

impl LoadedModel for LoadedMioCodec {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

pub struct LoadedSpeakerEncoder {
    pub wavlm: WavLM,
    pub global_encoder: GlobalEncoder,
}

impl LoadedModel for LoadedSpeakerEncoder {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

// ── LFM2 state ─────────────────────────────────────────────────────

pub struct CandleLfm2State {
    pub conv_states: Vec<Option<Tensor>>,
    pub kv_caches: Vec<Option<(Tensor, Tensor)>>,
    pub seq_offset: usize,
}

impl Lfm2State for CandleLfm2State {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn seq_offset(&self) -> usize { self.seq_offset }
}

// ── Backend trait implementation ───────────────────────────────────

impl Backend for CandleBackend {
    fn name(&self) -> &str {
        match self.device {
            Device::Cuda(_) => "candle-cuda",
            #[cfg(feature = "metal")]
            Device::Metal(_) => "candle-metal",
            _ => "candle-cpu",
        }
    }

    fn supported_components(&self) -> &[ModelComponent] {
        &[
            ModelComponent::SpeakerEncoder,
            ModelComponent::Lfm2,
            ModelComponent::MioCodec,
        ]
    }

    fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    fn load_model(
        &self,
        component: ModelComponent,
        _model_dir: &Path,
    ) -> Result<Box<dyn LoadedModel>, TtsError> {
        let paths = self.paths()?;

        match component {
            ModelComponent::Lfm2 => {
                let config_str = std::fs::read_to_string(&paths.lfm2_config)
                    .map_err(|e| TtsError::Model(format!("Read LFM2 config: {e}")))?;
                let config: Lfm2Config = serde_json::from_str(&config_str)
                    .map_err(|e| TtsError::Model(format!("Parse LFM2 config: {e}")))?;

                info!(
                    "LFM2 config: {} layers, {} hidden, {} vocab",
                    config.num_hidden_layers, config.hidden_size, config.vocab_size
                );

                let dtype = if self.device.is_cuda() { DType::BF16 } else { DType::F32 };
                info!("Loading LFM2 ({} files, {:?})...", paths.lfm2_safetensors.len(), dtype);
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &paths.lfm2_safetensors,
                        dtype,
                        &self.device,
                    )
                    .map_err(|e| TtsError::Model(format!("LFM2 safetensors: {e}")))?
                };
                let model = Lfm2ForCausalLM::load(vb, &config)
                    .map_err(|e| TtsError::Model(format!("LFM2 load: {e}")))?;
                info!("LFM2 loaded");

                Ok(Box::new(LoadedLfm2 { model, config }))
            }

            ModelComponent::MioCodec => {
                let config_str = std::fs::read_to_string(&paths.miocodec_config)
                    .map_err(|e| TtsError::Model(format!("Read MioCodec config: {e}")))?;
                let config = MioCodecConfig::from_yaml(&config_str);
                info!("MioCodec config: {}Hz, n_fft={}", config.sample_rate, config.n_fft);

                info!("Loading MioCodec...");
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        std::slice::from_ref(&paths.miocodec_safetensors),
                        DType::F32,
                        &self.device,
                    )
                    .map_err(|e| TtsError::Model(format!("MioCodec safetensors: {e}")))?
                };
                let model = MioCodecDecoder::load(vb, &config)
                    .map_err(|e| TtsError::Model(format!("MioCodec load: {e}")))?;
                info!("MioCodec loaded");

                Ok(Box::new(LoadedMioCodec { model, config }))
            }

            ModelComponent::SpeakerEncoder => {
                let miocodec_config_str = std::fs::read_to_string(&paths.miocodec_config)
                    .map_err(|e| TtsError::Model(format!("Read MioCodec config: {e}")))?;
                let miocodec_config = MioCodecConfig::from_yaml(&miocodec_config_str);

                info!("Loading WavLM...");
                let wavlm_vb = VarBuilder::from_pth(&paths.wavlm_pth, DType::F32, &self.device)
                    .map_err(|e| TtsError::Model(format!("WavLM pth: {e}")))?;
                let wavlm = WavLM::load(wavlm_vb, &miocodec_config.global_ssl_layers)
                    .map_err(|e| TtsError::Model(format!("WavLM load: {e}")))?;

                info!("Loading GlobalEncoder...");
                let ge_vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        std::slice::from_ref(&paths.miocodec_safetensors),
                        DType::F32,
                        &self.device,
                    )
                    .map_err(|e| TtsError::Model(format!("GlobalEncoder safetensors: {e}")))?
                };
                let global_encoder = GlobalEncoder::load(ge_vb.pp("global_encoder"))
                    .map_err(|e| TtsError::Model(format!("GlobalEncoder load: {e}")))?;

                info!("SpeakerEncoder loaded");
                Ok(Box::new(LoadedSpeakerEncoder {
                    wavlm,
                    global_encoder,
                }))
            }
        }
    }

    fn encode_speaker(
        &self,
        model: &dyn LoadedModel,
        wav_16k_mono: &[f32],
    ) -> Result<TensorData, TtsError> {
        let encoder = model
            .as_any()
            .downcast_ref::<LoadedSpeakerEncoder>()
            .ok_or_else(|| TtsError::Model("Expected LoadedSpeakerEncoder".into()))?;

        let len = wav_16k_mono.len();
        let waveform = Tensor::from_vec(wav_16k_mono.to_vec(), (1, len), &self.device)
            .map_err(|e| TtsError::Inference(format!("Wav tensor: {e}")))?;

        let ssl_features = encoder
            .wavlm
            .extract_global_features(&waveform)
            .map_err(|e| TtsError::Inference(format!("WavLM forward: {e}")))?;

        let embedding = encoder
            .global_encoder
            .forward(&ssl_features)
            .map_err(|e| TtsError::Inference(format!("GlobalEncoder forward: {e}")))?;

        // Return as Native tensor for zero-copy within candle backend
        Ok(TensorData::Native(Box::new(embedding)))
    }

    fn lfm2_prefill(
        &self,
        model: &dyn LoadedModel,
        input_ids: &[u32],
    ) -> Result<(Box<dyn Lfm2State>, Vec<f32>), TtsError> {
        let lfm2 = model
            .as_any()
            .downcast_ref::<LoadedLfm2>()
            .ok_or_else(|| TtsError::Model("Expected LoadedLfm2".into()))?;

        let input = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), &self.device)
            .map_err(|e| TtsError::Inference(format!("Input tensor: {e}")))?;

        let (mut conv_states, mut kv_caches) = lfm2.model.init_state();

        let logits = lfm2
            .model
            .forward(&input, &mut conv_states, &mut kv_caches, 0)
            .map_err(|e| TtsError::Inference(format!("LFM2 prefill: {e}")))?;

        let seq_len = input_ids.len();
        let last_logits = logits
            .narrow(1, seq_len - 1, 1)
            .and_then(|t| t.squeeze(1))
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| TtsError::Inference(format!("Extract logits: {e}")))?;

        let state = CandleLfm2State {
            conv_states,
            kv_caches,
            seq_offset: seq_len,
        };

        Ok((Box::new(state), last_logits))
    }

    fn lfm2_decode_step(
        &self,
        model: &dyn LoadedModel,
        state: &mut dyn Lfm2State,
        token: u32,
    ) -> Result<Vec<f32>, TtsError> {
        let lfm2 = model
            .as_any()
            .downcast_ref::<LoadedLfm2>()
            .ok_or_else(|| TtsError::Model("Expected LoadedLfm2".into()))?;

        let candle_state = state
            .as_any_mut()
            .downcast_mut::<CandleLfm2State>()
            .ok_or_else(|| TtsError::Model("Expected CandleLfm2State".into()))?;

        let input = Tensor::from_vec(vec![token], (1, 1), &self.device)
            .map_err(|e| TtsError::Inference(format!("Token tensor: {e}")))?;

        let logits = lfm2
            .model
            .forward(
                &input,
                &mut candle_state.conv_states,
                &mut candle_state.kv_caches,
                candle_state.seq_offset,
            )
            .map_err(|e| TtsError::Inference(format!("LFM2 decode step: {e}")))?;

        candle_state.seq_offset += 1;

        let logits_vec = logits
            .squeeze(1)
            .and_then(|t| t.squeeze(0))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| TtsError::Inference(format!("Extract logits: {e}")))?;

        Ok(logits_vec)
    }

    fn miocodec_decode(
        &self,
        model: &dyn LoadedModel,
        tokens: &[u32],
        speaker_embedding: &TensorData,
    ) -> Result<Vec<f32>, TtsError> {
        let miocodec = model
            .as_any()
            .downcast_ref::<LoadedMioCodec>()
            .ok_or_else(|| TtsError::Model("Expected LoadedMioCodec".into()))?;

        // Get speaker embedding tensor
        let spk_tensor = match speaker_embedding {
            TensorData::Native(any) => any
                .downcast_ref::<Tensor>()
                .ok_or_else(|| TtsError::Inference("Speaker embedding is not a candle Tensor".into()))?
                .clone(),
            TensorData::F32 { data, shape } => {
                Tensor::from_vec(data.clone(), shape.as_slice(), &self.device)
                    .map_err(|e| TtsError::Inference(format!("Speaker tensor: {e}")))?
            }
            _ => return Err(TtsError::Inference("Unexpected speaker embedding format".into())),
        };

        let num_tokens = tokens.len();
        let token_tensor = Tensor::from_vec(tokens.to_vec(), (num_tokens,), &self.device)
            .map_err(|e| TtsError::Inference(format!("Token tensor: {e}")))?;

        let waveform = miocodec
            .model
            .forward_wave(&token_tensor, &spk_tensor)
            .map_err(|e| TtsError::Inference(format!("MioCodec decode: {e}")))?;

        let pcm: Vec<f32> = waveform
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec1())
            .map_err(|e| TtsError::Inference(format!("MioCodec output: {e}")))?;

        Ok(pcm)
    }

    fn benchmark(
        &self,
        model: &dyn LoadedModel,
        component: ModelComponent,
    ) -> Result<BenchmarkResult, TtsError> {
        use std::time::Instant;

        const WARMUP: usize = 3;
        const MEASURED: usize = 5;

        // Run a tiny forward pass with dummy data to measure throughput.
        match component {
            ModelComponent::Lfm2 => {
                let lfm2 = model
                    .as_any()
                    .downcast_ref::<LoadedLfm2>()
                    .ok_or_else(|| TtsError::Model("Expected LoadedLfm2".into()))?;

                // Tiny prefill with 4 tokens
                let dummy_ids = Tensor::from_vec(vec![1u32; 4], (1, 4), &self.device)
                    .map_err(|e| TtsError::Inference(format!("Benchmark tensor: {e}")))?;

                for _ in 0..WARMUP {
                    let (mut cs, mut kv) = lfm2.model.init_state();
                    let _ = lfm2.model.forward(&dummy_ids, &mut cs, &mut kv, 0)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                }

                let mut total = std::time::Duration::ZERO;
                for _ in 0..MEASURED {
                    let (mut cs, mut kv) = lfm2.model.init_state();
                    let start = Instant::now();
                    let _ = lfm2.model.forward(&dummy_ids, &mut cs, &mut kv, 0)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                    total += start.elapsed();
                }

                Ok(BenchmarkResult {
                    component,
                    avg_duration: total / MEASURED as u32,
                    iterations: MEASURED,
                })
            }

            ModelComponent::MioCodec => {
                let miocodec = model
                    .as_any()
                    .downcast_ref::<LoadedMioCodec>()
                    .ok_or_else(|| TtsError::Model("Expected LoadedMioCodec".into()))?;

                // Dummy: 8 codec tokens + zero speaker embedding
                let dummy_tokens = Tensor::from_vec(vec![0u32; 8], (8,), &self.device)
                    .map_err(|e| TtsError::Inference(format!("Benchmark tensor: {e}")))?;
                let dummy_spk = Tensor::zeros((1, 128), DType::F32, &self.device)
                    .map_err(|e| TtsError::Inference(format!("Benchmark tensor: {e}")))?;

                for _ in 0..WARMUP {
                    let _ = miocodec.model.forward_wave(&dummy_tokens, &dummy_spk)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                }

                let mut total = std::time::Duration::ZERO;
                for _ in 0..MEASURED {
                    let start = Instant::now();
                    let _ = miocodec.model.forward_wave(&dummy_tokens, &dummy_spk)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                    total += start.elapsed();
                }

                Ok(BenchmarkResult {
                    component,
                    avg_duration: total / MEASURED as u32,
                    iterations: MEASURED,
                })
            }

            ModelComponent::SpeakerEncoder => {
                let encoder = model
                    .as_any()
                    .downcast_ref::<LoadedSpeakerEncoder>()
                    .ok_or_else(|| TtsError::Model("Expected LoadedSpeakerEncoder".into()))?;

                // Dummy: 1 second of silence at 16kHz
                let dummy_wav = Tensor::zeros((1, 16000), DType::F32, &self.device)
                    .map_err(|e| TtsError::Inference(format!("Benchmark tensor: {e}")))?;

                for _ in 0..WARMUP {
                    let ssl = encoder.wavlm.extract_global_features(&dummy_wav)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                    let _ = encoder.global_encoder.forward(&ssl)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                }

                let mut total = std::time::Duration::ZERO;
                for _ in 0..MEASURED {
                    let start = Instant::now();
                    let ssl = encoder.wavlm.extract_global_features(&dummy_wav)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                    let _ = encoder.global_encoder.forward(&ssl)
                        .map_err(|e| TtsError::Inference(format!("Benchmark forward: {e}")))?;
                    total += start.elapsed();
                }

                Ok(BenchmarkResult {
                    component,
                    avg_duration: total / MEASURED as u32,
                    iterations: MEASURED,
                })
            }
        }
    }
}
