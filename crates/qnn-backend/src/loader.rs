//! TFLite model loading and QNN delegate setup.
//!
//! On Android, loads a `.tflite` model and configures the QNN delegate to
//! target the Hexagon NPU (HTP backend).  On other platforms, provides stub
//! types that still compile.

use std::path::{Path, PathBuf};

use any_miotts_core::error::TtsError;

/// Represents a loaded TFLite model with QNN delegate.
///
/// On Android this wraps a `tflitec::interpreter::Interpreter`.
/// On non-Android platforms this is a stub.
pub struct TfLiteModel {
    /// Path to the .tflite file (for diagnostics).
    #[allow(dead_code)]
    pub(crate) model_path: PathBuf,

    /// The TFLite interpreter handle (Android only).
    #[cfg(target_os = "android")]
    pub(crate) interpreter: tflitec::interpreter::Interpreter,
}

// SAFETY: TFLite Interpreter is thread-safe for single-threaded inference
// once configured. We ensure exclusive access via &mut in inference calls.
unsafe impl Send for TfLiteModel {}
unsafe impl Sync for TfLiteModel {}

/// Locate the MioCodec TFLite model in a directory.
///
/// Looks for a `.tflite` file within the given model directory.
pub fn find_tflite_model(model_dir: &Path) -> Result<PathBuf, TtsError> {
    // Check if model_dir itself is a .tflite file
    if let Some(ext) = model_dir.extension() {
        if ext.to_string_lossy() == "tflite" {
            return Ok(model_dir.to_path_buf());
        }
    }

    // Search for a .tflite file in the directory
    let entries = std::fs::read_dir(model_dir)
        .map_err(|e| TtsError::Model(format!("Read dir {}: {e}", model_dir.display())))?;

    for entry in entries {
        let entry = entry.map_err(|e| TtsError::Model(format!("Dir entry: {e}")))?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext.to_string_lossy() == "tflite" {
                return Ok(path);
            }
        }
    }

    Err(TtsError::Model(format!(
        "No .tflite model found in {}",
        model_dir.display()
    )))
}

/// Load a TFLite model with QNN delegate for Hexagon NPU inference.
///
/// On Android, this creates a TFLite interpreter, attaches the QNN delegate
/// targeting the HTP (Hexagon Tensor Processor) backend, and allocates tensors.
/// On non-Android platforms, this always returns an error.
#[cfg(target_os = "android")]
pub fn load_model(model_path: &Path) -> Result<TfLiteModel, TtsError> {
    use tflitec::interpreter::Interpreter;
    use tflitec::model::Model;
    use tracing::info;

    let path_str = model_path
        .to_str()
        .ok_or_else(|| TtsError::Model("Model path is not valid UTF-8".into()))?;

    info!("Loading TFLite model from {}", model_path.display());

    let model = Model::new(path_str)
        .map_err(|e| TtsError::Model(format!("Failed to load TFLite model: {e:?}")))?;

    let mut builder = Interpreter::builder();
    builder.model(&model);

    // Try to add QNN delegate for Hexagon NPU acceleration.
    // The QNN delegate is loaded as a shared library (libQnnHtp.so).
    // If QNN is not available, fall back to CPU inference.
    let qnn_delegate_path = "/data/local/tmp/libQnnHtp.so";
    if Path::new(qnn_delegate_path).exists() {
        info!("QNN HTP delegate found at {qnn_delegate_path}");
        // QNN delegate is configured via external delegate options.
        // TFLite will load the delegate shared library and route ops to the NPU.
        builder.add_external_delegate(
            qnn_delegate_path,
            &[("backend_type", "htp"), ("htp_performance_mode", "burst")],
        );
        info!("QNN HTP delegate attached");
    } else {
        info!("QNN HTP delegate not found, using CPU fallback");
    }

    let interpreter = builder
        .build()
        .map_err(|e| TtsError::Model(format!("Failed to build TFLite interpreter: {e:?}")))?;

    interpreter
        .allocate_tensors()
        .map_err(|e| TtsError::Model(format!("Failed to allocate tensors: {e:?}")))?;

    info!("TFLite model loaded successfully");

    Ok(TfLiteModel {
        model_path: model_path.to_path_buf(),
        interpreter,
    })
}

/// Stub loader for non-Android platforms.
#[cfg(not(target_os = "android"))]
pub fn load_model(model_path: &Path) -> Result<TfLiteModel, TtsError> {
    Err(TtsError::Unsupported(format!(
        "QNN/TFLite is only available on Android (tried to load {})",
        model_path.display()
    )))
}
