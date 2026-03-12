//! CoreML model loading logic.
//!
//! Handles loading `.mlpackage` (source) or `.mlmodelc` (compiled) CoreML
//! models from disk.  On Apple platforms, this uses the CoreML framework
//! to compile and instantiate the model.  On other platforms, this module
//! provides stub types.

use std::path::{Path, PathBuf};

use any_miotts_core::error::TtsError;

/// Represents a loaded CoreML model.
///
/// On Apple platforms this wraps an `MLModel` instance.
/// On non-Apple platforms this is a zero-size stub that cannot be constructed.
pub struct CoreMlModel {
    /// Path to the model (for diagnostics).
    #[allow(dead_code)]
    pub(crate) model_path: PathBuf,

    /// The loaded MLModel handle (Apple only).
    #[cfg(target_vendor = "apple")]
    pub(crate) model: objc2_core_ml::MLModel,
}

// SAFETY: MLModel is thread-safe for prediction once loaded.
unsafe impl Send for CoreMlModel {}
unsafe impl Sync for CoreMlModel {}

/// Locate the MioCodec CoreML model in a directory.
///
/// Looks for either a `.mlpackage` or `.mlmodelc` (pre-compiled) directory
/// within the given model directory.  Prefers `.mlmodelc` if both exist.
pub fn find_coreml_model(model_dir: &Path) -> Result<PathBuf, TtsError> {
    // First, check if model_dir itself is a .mlpackage or .mlmodelc
    if let Some(ext) = model_dir.extension() {
        let ext = ext.to_string_lossy();
        if ext == "mlmodelc" || ext == "mlpackage" {
            return Ok(model_dir.to_path_buf());
        }
    }

    // Look for .mlmodelc first (pre-compiled, faster to load)
    let entries = std::fs::read_dir(model_dir)
        .map_err(|e| TtsError::Model(format!("Read dir {}: {e}", model_dir.display())))?;

    let mut mlpackage_path: Option<PathBuf> = None;

    for entry in entries {
        let entry = entry.map_err(|e| TtsError::Model(format!("Dir entry: {e}")))?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy();
            if ext == "mlmodelc" {
                return Ok(path);
            }
            if ext == "mlpackage" {
                mlpackage_path = Some(path);
            }
        }
    }

    // Fall back to .mlpackage
    if let Some(path) = mlpackage_path {
        return Ok(path);
    }

    Err(TtsError::Model(format!(
        "No .mlpackage or .mlmodelc found in {}",
        model_dir.display()
    )))
}

/// Load a CoreML model from a `.mlpackage` or `.mlmodelc` path.
///
/// On Apple platforms, this compiles the model (if `.mlpackage`) and loads it.
/// On non-Apple platforms, this always returns an error.
#[cfg(target_vendor = "apple")]
pub fn load_model(model_path: &Path) -> Result<CoreMlModel, TtsError> {
    use objc2_core_ml::{MLModel, MLModelConfiguration};
    use objc2_foundation::NSURL;
    use tracing::info;

    let path_str = model_path
        .to_str()
        .ok_or_else(|| TtsError::Model("Model path is not valid UTF-8".into()))?;

    info!("Loading CoreML model from {}", model_path.display());

    let config = unsafe { MLModelConfiguration::new() };

    // Prefer ANE, fall back to CPU+GPU
    // computeUnits = .all allows CoreML to use ANE when possible
    unsafe {
        config.setComputeUnits(objc2_core_ml::MLComputeUnits::All);
    }

    let url = unsafe {
        NSURL::fileURLWithPath(&objc2_foundation::NSString::from_str(path_str))
    };

    let model = unsafe {
        MLModel::modelWithContentsOfURL_configuration_error(&url, &config)
    }
    .map_err(|e| TtsError::Model(format!("Failed to load CoreML model: {e}")))?;

    info!("CoreML model loaded successfully");

    Ok(CoreMlModel {
        model_path: model_path.to_path_buf(),
        model,
    })
}

/// Stub loader for non-Apple platforms.
#[cfg(not(target_vendor = "apple"))]
pub fn load_model(model_path: &Path) -> Result<CoreMlModel, TtsError> {
    Err(TtsError::Unsupported(format!(
        "CoreML is only available on Apple platforms (tried to load {})",
        model_path.display()
    )))
}
