//! Direct Safetensors GPU inference through WGPU.

pub mod error;
mod model;
mod runtime;
pub mod weights;

pub use crate::models::ModelSize;
pub use error::{Error, Result};
pub use model::{Detector, ModelOutput, Recognizer};
pub use runtime::{Gpu, GpuImage, GpuInfo, ImagePreprocess};
pub use weights::{F32Tensor, Weights};
