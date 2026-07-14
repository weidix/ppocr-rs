//! Direct Safetensors CPU inference.

mod backend;
mod kernels;
mod model;
mod ops;
mod tensor;
mod weights;

pub use model::{CpuOptions, Detector, ModelSize, Recognizer};
pub use tensor::Tensor;
