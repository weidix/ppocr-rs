//! Native PP-OCRv6 inference for Rust.
//!
//! The [`ModelStore`] keeps the repository's pinned model packages available
//! locally, and the CPU feature exposes [`OcrEngine`] for image-to-text OCR.
//! Lower-level detector, recognizer, and postprocessing APIs remain available
//! through the feature-gated backend modules.

pub mod models;
pub mod ocr;
#[cfg(feature = "cpu")]
mod preprocess;

pub use models::{ModelKind, ModelPaths, ModelSize, ModelStore, OcrModelPaths};
pub use ocr::DetectorPostprocessOptions;

#[cfg(feature = "cpu")]
pub use cpu::CpuOptions;
#[cfg(feature = "cpu")]
pub use ocr::{OcrEngine, OcrLine, OcrOptions, OcrResult};

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;
