//! llama.cpp backend for any-miotts.
//!
//! This backend supports **only** the LFM2 model component, loading GGUF
//! quantised weights via llama.cpp.  It is designed for Android deployment
//! where the OpenCL backend can target Qualcomm Adreno GPUs.

pub mod backend;
pub mod discovery;
