pub mod ocr;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cpu_onnx")]
pub mod cpu_onnx;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "burn")]
pub mod burn_runtime;

#[cfg(feature = "candle")]
pub mod model;

#[cfg(feature = "candle")]
pub mod preprocess;

#[cfg(feature = "training")]
pub mod training;
