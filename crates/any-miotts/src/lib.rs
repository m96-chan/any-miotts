//! any-miotts: Multi-backend TTS engine with automatic device selection.
//!
//! Wraps candle-miotts and (future) llama.cpp / CoreML / QNN backends,
//! automatically selecting the best device for each model component.

pub use any_miotts_core::backend::{Backend, ModelComponent, TensorData};
pub use any_miotts_core::device::{DeviceInfo, DeviceKind};
pub use any_miotts_core::engine::{SynthesisEvent, TtsEngine};
pub use any_miotts_core::error::TtsError;
pub use any_miotts_core::sampling::GenerateParams;
pub use any_miotts_core::sentence::SentenceSplitter;

pub use any_miotts_candle::backend::CandleBackend;
pub use any_miotts_candle::discovery::discover_devices;

#[cfg(feature = "llamacpp")]
pub use any_miotts_llamacpp::backend::{LlamaCppBackend, LlamaCppConfig};
#[cfg(feature = "llamacpp")]
pub use any_miotts_llamacpp::discovery as llamacpp_discovery;

#[cfg(feature = "coreml")]
pub use any_miotts_coreml::backend::CoreMlBackend;
#[cfg(feature = "coreml")]
pub use any_miotts_coreml::discovery as coreml_discovery;

#[cfg(feature = "qnn")]
pub use any_miotts_qnn::backend::QnnBackend;
#[cfg(feature = "qnn")]
pub use any_miotts_qnn::discovery as qnn_discovery;

use std::path::Path;

use tracing::info;

/// Initialize the TTS engine with automatic backend/device selection.
///
/// 1. Discovers available devices (CUDA, Metal, CPU)
/// 2. Creates a CandleBackend for each device
/// 3. Downloads models if needed
/// 4. Auto-assigns components to backends using hard rules + preference scoring
/// 5. Loads all model components
/// 6. Computes speaker embedding from reference wav
pub async fn initialize(reference_wav: &Path) -> Result<TtsEngine, TtsError> {
    info!("Initializing any-miotts engine...");

    // Discover all available devices
    let discovered = discover_devices();
    info!("Discovered {} device(s):", discovered.len());
    for (info, _) in &discovered {
        info!("  - {}", info);
    }

    // Create a CandleBackend for each device and download models
    let mut backends: Vec<Box<dyn Backend>> = Vec::new();
    let mut first_paths: Option<any_miotts_candle::backend::CandleModelPaths> = None;

    for (device_info, device) in discovered {
        let mut backend = CandleBackend::new(device_info, device);
        backend.ensure_models().await?;
        if first_paths.is_none() {
            first_paths = backend.paths_ref().cloned();
        }
        backends.push(Box::new(backend));
    }

    // When the llamacpp feature is enabled, add a LlamaCppBackend for LFM2.
    // This allows the scheduler to prefer llama.cpp for LFM2 on devices
    // where it performs better (e.g. Android with OpenCL GPU).
    #[cfg(feature = "llamacpp")]
    {
        let llamacpp = LlamaCppBackend::with_defaults();
        info!(
            "llama.cpp backend available: {} ({})",
            llamacpp.name(),
            llamacpp.device_info()
        );
        backends.push(Box::new(llamacpp));
    }

    // When the coreml feature is enabled, add a CoreMlBackend for MioCodec.
    // On Apple Silicon this offloads MioCodec decoding to the Apple Neural
    // Engine (ANE) via CoreML, freeing the GPU for LFM2.
    #[cfg(feature = "coreml")]
    {
        let coreml = CoreMlBackend::with_defaults();
        info!(
            "CoreML backend available: {} ({})",
            coreml.name(),
            coreml.device_info()
        );
        backends.push(Box::new(coreml));
    }

    // When the qnn feature is enabled, add a QnnBackend for MioCodec.
    // On Qualcomm SoCs this offloads MioCodec decoding to the Hexagon NPU
    // via TFLite with the QNN delegate, freeing the GPU for LFM2.
    #[cfg(feature = "qnn")]
    {
        let qnn = QnnBackend::with_defaults();
        info!(
            "QNN backend available: {} ({})",
            qnn.name(),
            qnn.device_info()
        );
        backends.push(Box::new(qnn));
    }

    let paths = first_paths
        .ok_or_else(|| TtsError::Model("No backends created".into()))?;

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(&paths.lfm2_tokenizer)
        .map_err(|e| TtsError::Model(format!("Tokenizer: {e}")))?;
    info!("Tokenizer loaded ({} tokens)", tokenizer.get_vocab_size(true));

    // Read LFM2 config for eos_token_id
    let config_str = std::fs::read_to_string(&paths.lfm2_config)
        .map_err(|e| TtsError::Model(format!("Read LFM2 config: {e}")))?;
    let lfm2_config: candle_miotts::models::config::Lfm2Config =
        serde_json::from_str(&config_str)
            .map_err(|e| TtsError::Model(format!("Parse LFM2 config: {e}")))?;

    // Read MioCodec config for sample_rate
    let miocodec_config_str = std::fs::read_to_string(&paths.miocodec_config)
        .map_err(|e| TtsError::Model(format!("Read MioCodec config: {e}")))?;
    let miocodec_config = candle_miotts::models::config::MioCodecConfig::from_yaml(&miocodec_config_str);

    let eos_token_id = lfm2_config.eos_token_id;
    let sample_rate = miocodec_config.sample_rate;

    // Build engine (auto-assigns components to best backends)
    let reference_wav = reference_wav.to_path_buf();
    let engine = tokio::task::spawn_blocking(move || {
        TtsEngine::build(
            backends,
            &reference_wav,
            tokenizer,
            eos_token_id,
            sample_rate,
        )
    })
    .await
    .map_err(|e| TtsError::Model(format!("Engine init join error: {e}")))?;

    engine
}
