//! QNN/Hexagon NPU backend for any-miotts.
//!
//! This backend supports **only** the MioCodec model component, running on
//! Qualcomm's Hexagon NPU via TFLite with the QNN delegate.  MioCodec is a
//! fixed-shape single forward pass, making it ideal for NPU acceleration.
//!
//! On non-Android platforms this crate compiles as a no-op: the backend type
//! is still exported but all methods return `Err(Unsupported)`.

pub mod backend;
pub mod discovery;
pub mod loader;
