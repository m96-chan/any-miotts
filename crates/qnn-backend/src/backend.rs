//! [`Backend`] implementation for QNN, targeting MioCodec on Qualcomm Hexagon NPU.
//!
//! This backend supports **only** [`ModelComponent::MioCodec`].  It loads a
//! TFLite `.tflite` model and runs inference via the QNN delegate, which
//! automatically targets the Hexagon NPU (HTP) when available.

use std::any::Any;
use std::path::Path;
use std::time::{Duration, Instant};

use tracing::info;

use any_miotts_core::backend::{
    Backend, BenchmarkResult, Lfm2State, LoadedModel, ModelComponent, TensorData,
};
use any_miotts_core::device::DeviceInfo;
use any_miotts_core::error::TtsError;

use crate::loader::TfLiteModel;

// -- Loaded model wrapper --------------------------------------------------

/// Wrapper around a TFLite model that implements [`LoadedModel`].
pub struct LoadedQnnMioCodec {
    #[allow(dead_code)]
    pub(crate) model: TfLiteModel,
}

impl LoadedModel for LoadedQnnMioCodec {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// -- Backend ---------------------------------------------------------------

/// QNN backend for MioCodec decoding on Qualcomm Hexagon NPU.
///
/// This backend supports **only** [`ModelComponent::MioCodec`].  Speaker
/// encoding and LFM2 generation must be handled by another backend
/// (e.g. llama.cpp with OpenCL).
pub struct QnnBackend {
    device_info: DeviceInfo,
}

impl QnnBackend {
    /// Create a new QNN backend with the given device info.
    pub fn new(device_info: DeviceInfo) -> Self {
        Self { device_info }
    }

    /// Create a backend using auto-discovered device.
    pub fn with_defaults() -> Self {
        let device_info = crate::discovery::discover_device();
        Self { device_info }
    }
}

impl Backend for QnnBackend {
    fn name(&self) -> &str {
        "qnn-hexagon"
    }

    fn supported_components(&self) -> &[ModelComponent] {
        &[ModelComponent::MioCodec]
    }

    fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    fn load_model(
        &self,
        component: ModelComponent,
        model_dir: &Path,
    ) -> Result<Box<dyn LoadedModel>, TtsError> {
        if component != ModelComponent::MioCodec {
            return Err(TtsError::Unsupported(format!(
                "{}: only MioCodec is supported, got {component}",
                self.name()
            )));
        }

        let model_path = crate::loader::find_tflite_model(model_dir)?;
        info!("Loading TFLite MioCodec from {}", model_path.display());

        let model = crate::loader::load_model(&model_path)?;
        Ok(Box::new(LoadedQnnMioCodec { model }))
    }

    // -- MioCodec ----------------------------------------------------------

    fn miocodec_decode(
        &self,
        model: &dyn LoadedModel,
        tokens: &[u32],
        speaker_embedding: &TensorData,
    ) -> Result<Vec<f32>, TtsError> {
        let loaded = model
            .as_any()
            .downcast_ref::<LoadedQnnMioCodec>()
            .ok_or_else(|| TtsError::Model("Expected LoadedQnnMioCodec".into()))?;

        let spk_data = match speaker_embedding {
            TensorData::F32 { data, shape } => {
                if shape.iter().product::<usize>() != 128 {
                    return Err(TtsError::Inference(format!(
                        "Speaker embedding must be 128-dim, got shape {shape:?}"
                    )));
                }
                data
            }
            _ => {
                return Err(TtsError::Inference(
                    "Speaker embedding must be F32 TensorData".into(),
                ));
            }
        };

        predict_tflite(loaded, tokens, spk_data)
    }

    // -- Unsupported operations --------------------------------------------

    fn encode_speaker(
        &self,
        _model: &dyn LoadedModel,
        _wav_16k_mono: &[f32],
    ) -> Result<TensorData, TtsError> {
        Err(TtsError::Unsupported(format!(
            "{}: encode_speaker not supported (MioCodec only backend)",
            self.name()
        )))
    }

    fn lfm2_prefill(
        &self,
        _model: &dyn LoadedModel,
        _input_ids: &[u32],
    ) -> Result<(Box<dyn Lfm2State>, Vec<f32>), TtsError> {
        Err(TtsError::Unsupported(format!(
            "{}: lfm2_prefill not supported (MioCodec only backend)",
            self.name()
        )))
    }

    fn lfm2_decode_step(
        &self,
        _model: &dyn LoadedModel,
        _state: &mut dyn Lfm2State,
        _token: u32,
    ) -> Result<Vec<f32>, TtsError> {
        Err(TtsError::Unsupported(format!(
            "{}: lfm2_decode_step not supported (MioCodec only backend)",
            self.name()
        )))
    }

    // -- Benchmark ---------------------------------------------------------

    fn benchmark(
        &self,
        model: &dyn LoadedModel,
        component: ModelComponent,
    ) -> Result<BenchmarkResult, TtsError> {
        if component != ModelComponent::MioCodec {
            return Err(TtsError::Unsupported(format!(
                "{}: benchmark only supported for MioCodec",
                self.name()
            )));
        }

        const WARMUP: usize = 2;
        const MEASURED: usize = 5;

        // Dummy inputs for benchmarking.
        let dummy_tokens: Vec<u32> = vec![0; 100];
        let dummy_spk = TensorData::F32 {
            data: vec![0.0f32; 128],
            shape: vec![128],
        };

        // Warmup
        for _ in 0..WARMUP {
            let _ = self.miocodec_decode(model, &dummy_tokens, &dummy_spk)?;
        }

        // Measured
        let mut total = Duration::ZERO;
        for _ in 0..MEASURED {
            let start = Instant::now();
            let _ = self.miocodec_decode(model, &dummy_tokens, &dummy_spk)?;
            total += start.elapsed();
        }

        Ok(BenchmarkResult {
            component,
            avg_duration: total / MEASURED as u32,
            iterations: MEASURED,
        })
    }
}

// -- TFLite prediction -----------------------------------------------------

/// Run MioCodec inference via TFLite with QNN delegate (Android only).
///
/// Input 0: codec_tokens (int32, [1, seq_len])
/// Input 1: speaker_embedding (float32, [1, 128])
/// Output 0: waveform (float32, [1, audio_len])
#[cfg(target_os = "android")]
fn predict_tflite(
    loaded: &LoadedQnnMioCodec,
    tokens: &[u32],
    speaker_embedding: &[f32],
) -> Result<Vec<f32>, TtsError> {
    let interp = &loaded.model.interpreter;

    // Prepare codec tokens as int32 input (input 0)
    let token_data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let token_bytes: &[u8] = bytemuck_cast_slice(&token_data);
    interp
        .copy(token_bytes, 0)
        .map_err(|e| TtsError::Inference(format!("Set input 0 (tokens): {e:?}")))?;

    // Prepare speaker embedding as float32 input (input 1)
    let spk_bytes: &[u8] = bytemuck_cast_slice(speaker_embedding);
    interp
        .copy(spk_bytes, 1)
        .map_err(|e| TtsError::Inference(format!("Set input 1 (speaker_embedding): {e:?}")))?;

    // Run inference
    interp
        .invoke()
        .map_err(|e| TtsError::Inference(format!("TFLite invoke failed: {e:?}")))?;

    // Read output waveform (output 0)
    let output_tensor = interp
        .output(0)
        .map_err(|e| TtsError::Inference(format!("Get output 0: {e:?}")))?;

    let output_data = output_tensor.data::<f32>();
    Ok(output_data.to_vec())
}

/// Reinterpret a typed slice as a byte slice (for TFLite input copying).
#[cfg(target_os = "android")]
fn bytemuck_cast_slice<T: Copy>(data: &[T]) -> &[u8] {
    let ptr = data.as_ptr() as *const u8;
    let len = data.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// Stub prediction for non-Android platforms.
#[cfg(not(target_os = "android"))]
fn predict_tflite(
    _loaded: &LoadedQnnMioCodec,
    _tokens: &[u32],
    _speaker_embedding: &[f32],
) -> Result<Vec<f32>, TtsError> {
    Err(TtsError::Unsupported(
        "QNN/TFLite inference is only available on Android".into(),
    ))
}
