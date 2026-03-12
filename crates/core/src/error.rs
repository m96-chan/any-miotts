use thiserror::Error;

#[derive(Debug, Error)]
pub enum TtsError {
    #[error("Model error: {0}")]
    Model(String),
    #[error("Inference error: {0}")]
    Inference(String),
    #[error("Download error: {0}")]
    Download(String),
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    #[error("No backend available for {0:?}")]
    NoBackend(crate::backend::ModelComponent),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}
