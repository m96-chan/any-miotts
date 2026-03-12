//! CoreML/ANE backend for any-miotts.
//!
//! This backend supports **only** the MioCodec model component, running on
//! Apple's Neural Engine (ANE) via CoreML.  MioCodec is a fixed-shape single
//! forward pass, making it ideal for ANE acceleration.
//!
//! On non-Apple platforms this crate compiles as a no-op: the backend type
//! is still exported but all methods return `Err(Unsupported)`.

pub mod backend;
pub mod discovery;
pub mod loader;
