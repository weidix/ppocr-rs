//! Direct Safetensors CPU inference.

mod arena;
mod backend;
mod kernels;
mod model;
mod ops;
mod tensor;
mod weights;
#[cfg(target_os = "windows")]
mod windows;

pub use crate::models::ModelSize;
pub use model::{CpuOptions, Detector, Recognizer};
pub use tensor::Tensor;
