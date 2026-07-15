//! Direct Safetensors CPU inference.

mod backend;
mod kernels;
mod model;
mod ops;
mod tensor;
mod weights;
#[cfg(target_os = "windows")]
mod windows;

pub use model::{CpuOptions, Detector, ModelSize, Recognizer};
pub use tensor::Tensor;
