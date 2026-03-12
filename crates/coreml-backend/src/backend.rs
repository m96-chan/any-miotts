//! [`Backend`] implementation for CoreML, targeting MioCodec on Apple ANE.
//!
//! This backend supports **only** [`ModelComponent::MioCodec`].  It loads a
//! CoreML `.mlpackage` or `.mlmodelc` model and runs inference via the CoreML
//! framework, which automatically targets the Apple Neural Engine (ANE) when
//! available.

use std::any::Any;
use std::path::Path;
use std::time::{Duration, Instant};

use tracing::info;

use any_miotts_core::backend::{
    Backend, BenchmarkResult, Lfm2State, LoadedModel, ModelComponent, TensorData,
};
use any_miotts_core::device::DeviceInfo;
use any_miotts_core::error::TtsError;

use crate::loader::CoreMlModel;

// ── Loaded model wrapper ──────────────────────────────────────────────

/// Wrapper around a CoreML model that implements [`LoadedModel`].
pub struct LoadedCoreMlMioCodec {
    #[allow(dead_code)]
    pub(crate) model: CoreMlModel,
}

impl LoadedModel for LoadedCoreMlMioCodec {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Backend ───────────────────────────────────────────────────────────

/// CoreML backend for MioCodec decoding on Apple Neural Engine.
///
/// This backend supports **only** [`ModelComponent::MioCodec`].  Speaker
/// encoding and LFM2 generation must be handled by another backend
/// (e.g. candle with Metal).
pub struct CoreMlBackend {
    device_info: DeviceInfo,
}

impl CoreMlBackend {
    /// Create a new CoreML backend with the given device info.
    pub fn new(device_info: DeviceInfo) -> Self {
        Self { device_info }
    }

    /// Create a backend using auto-discovered device.
    pub fn with_defaults() -> Self {
        let device_info = crate::discovery::discover_device();
        Self { device_info }
    }
}

impl Backend for CoreMlBackend {
    fn name(&self) -> &str {
        "coreml-ane"
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

        let model_path = crate::loader::find_coreml_model(model_dir)?;
        info!("Loading CoreML MioCodec from {}", model_path.display());

        let model = crate::loader::load_model(&model_path)?;
        Ok(Box::new(LoadedCoreMlMioCodec { model }))
    }

    // ── MioCodec ──────────────────────────────────────────────────────

    fn miocodec_decode(
        &self,
        model: &dyn LoadedModel,
        tokens: &[u32],
        speaker_embedding: &TensorData,
    ) -> Result<Vec<f32>, TtsError> {
        let _loaded = model
            .as_any()
            .downcast_ref::<LoadedCoreMlMioCodec>()
            .ok_or_else(|| TtsError::Model("Expected LoadedCoreMlMioCodec".into()))?;

        let _spk_data = match speaker_embedding {
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

        predict_coreml(_loaded, tokens, _spk_data)
    }

    // ── Unsupported operations ────────────────────────────────────────

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

    // ── Benchmark ─────────────────────────────────────────────────────

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

// ── CoreML prediction ─────────────────────────────────────────────────

/// Run MioCodec inference via CoreML (Apple platforms only).
#[cfg(target_vendor = "apple")]
fn predict_coreml(
    loaded: &LoadedCoreMlMioCodec,
    tokens: &[u32],
    speaker_embedding: &[f32],
) -> Result<Vec<f32>, TtsError> {
    use objc2_core_ml::{MLDictionaryFeatureProvider, MLFeatureValue, MLMultiArray};
    use objc2_foundation::{NSDictionary, NSNumber, NSString};

    // Create MLMultiArray for codec tokens (int32, shape [1, seq_len])
    let seq_len = tokens.len();
    let token_shape: Vec<_> = vec![
        unsafe { NSNumber::numberWithUnsignedInteger(1) },
        unsafe { NSNumber::numberWithUnsignedInteger(seq_len) },
    ];
    let token_shape_arr = objc2_foundation::NSArray::from_retained_slice(&token_shape);

    let token_array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &token_shape_arr,
            objc2_core_ml::MLMultiArrayDataType::Int32,
        )
    }
    .map_err(|e| TtsError::Inference(format!("Create token MLMultiArray: {e}")))?;

    // Fill token data
    for (i, &tok) in tokens.iter().enumerate() {
        let idx = vec![
            unsafe { NSNumber::numberWithUnsignedInteger(0) },
            unsafe { NSNumber::numberWithUnsignedInteger(i) },
        ];
        let idx_arr = objc2_foundation::NSArray::from_retained_slice(&idx);
        unsafe {
            token_array.setObject_atIndexedSubscript(
                &NSNumber::numberWithInt(tok as i32),
                &idx_arr,
            );
        }
    }

    // Create MLMultiArray for speaker embedding (float32, shape [1, 128])
    let spk_shape: Vec<_> = vec![
        unsafe { NSNumber::numberWithUnsignedInteger(1) },
        unsafe { NSNumber::numberWithUnsignedInteger(128) },
    ];
    let spk_shape_arr = objc2_foundation::NSArray::from_retained_slice(&spk_shape);

    let spk_array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &spk_shape_arr,
            objc2_core_ml::MLMultiArrayDataType::Float32,
        )
    }
    .map_err(|e| TtsError::Inference(format!("Create speaker MLMultiArray: {e}")))?;

    // Fill speaker embedding data
    for (i, &val) in speaker_embedding.iter().enumerate() {
        let idx = vec![
            unsafe { NSNumber::numberWithUnsignedInteger(0) },
            unsafe { NSNumber::numberWithUnsignedInteger(i) },
        ];
        let idx_arr = objc2_foundation::NSArray::from_retained_slice(&idx);
        unsafe {
            spk_array.setObject_atIndexedSubscript(
                &NSNumber::numberWithFloat(val),
                &idx_arr,
            );
        }
    }

    // Create feature values
    let token_feature = unsafe {
        MLFeatureValue::featureValueWithMultiArray(&token_array)
    };
    let spk_feature = unsafe {
        MLFeatureValue::featureValueWithMultiArray(&spk_array)
    };

    // Create feature provider
    let keys = vec![
        NSString::from_str("codec_tokens"),
        NSString::from_str("speaker_embedding"),
    ];
    let values = vec![token_feature, spk_feature];

    let dict = NSDictionary::from_retained_objects(&keys, &values);
    let provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &dict,
        )
    }
    .map_err(|e| TtsError::Inference(format!("Create feature provider: {e}")))?;

    // Run prediction
    let result = unsafe {
        loaded.model.model.predictionFromFeatures_error(&provider)
    }
    .map_err(|e| TtsError::Inference(format!("CoreML prediction failed: {e}")))?;

    // Extract waveform output
    let output_name = NSString::from_str("waveform");
    let output_feature = unsafe { result.featureValueForName(&output_name) }
        .ok_or_else(|| TtsError::Inference("No 'waveform' output in CoreML result".into()))?;

    let output_array = unsafe { output_feature.multiArrayValue() }
        .ok_or_else(|| TtsError::Inference("Waveform output is not a multi-array".into()))?;

    // Read output samples
    let count = unsafe { output_array.count() };
    let mut waveform = Vec::with_capacity(count);
    for i in 0..count {
        let idx = vec![unsafe { NSNumber::numberWithUnsignedInteger(i) }];
        let idx_arr = objc2_foundation::NSArray::from_retained_slice(&idx);
        let val = unsafe { output_array.objectAtIndexedSubscript(&idx_arr) };
        waveform.push(unsafe { val.floatValue() });
    }

    Ok(waveform)
}

/// Stub prediction for non-Apple platforms.
#[cfg(not(target_vendor = "apple"))]
fn predict_coreml(
    _loaded: &LoadedCoreMlMioCodec,
    _tokens: &[u32],
    _speaker_embedding: &[f32],
) -> Result<Vec<f32>, TtsError> {
    Err(TtsError::Unsupported(
        "CoreML inference is only available on Apple platforms".into(),
    ))
}
