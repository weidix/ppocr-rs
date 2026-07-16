//! Native PP-OCRv6 inference for Rust.
//!
//! The [`ModelStore`] keeps the repository's pinned model packages available
//! locally, and [`OcrEngine`] exposes image-to-text OCR on the selected CPU or
//! GPU backend. Lower-level detector, recognizer, and postprocessing APIs
//! remain available through the feature-gated backend modules.

pub mod models;
pub mod ocr;
mod pixels;
#[cfg(any(feature = "cpu", feature = "gpu"))]
mod preprocess;

pub use models::{ModelAccess, ModelKind, ModelPaths, ModelSize, ModelStore, OcrModelPaths};
pub use ocr::{DetectorPostprocessOptions, OcrBackend, OcrEngine, OcrLine, OcrOptions, OcrResult};
pub use pixels::RgbImage;

#[cfg(feature = "cpu")]
pub use cpu::CpuOptions;
#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;
