pub mod ocr;

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;
