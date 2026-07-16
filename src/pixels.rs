//! Minimal owned RGB image storage used by the OCR pipeline.
//!
//! The `image` crate is intentionally confined to file decoding here.  All
//! resizing, rectification, and normalization are implemented by the selected
//! inference backend so the GPU path never round-trips preprocessed pixels.

use anyhow::{Context, Result, ensure};
use std::path::Path;

/// An interleaved, row-major RGB8 image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl RgbImage {
    /// Creates an RGB image from interleaved RGB8 pixels.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        ensure!(width > 0 && height > 0, "image dimensions must be non-zero");
        let expected = usize::try_from(width)
            .context("image width does not fit usize")?
            .checked_mul(usize::try_from(height).context("image height does not fit usize")?)
            .and_then(|pixels| pixels.checked_mul(3))
            .context("image dimensions overflow")?;
        ensure!(
            pixels.len() == expected,
            "RGB image has {} bytes; expected {expected} for {width}x{height}",
            pixels.len()
        );
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Decodes a PNG or JPEG image into RGB8 storage.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = ::image::ImageReader::open(path)
            .with_context(|| format!("open image {}", path.display()))?
            .decode()
            .with_context(|| format!("decode image {}", path.display()))?
            .to_rgb8();
        Self::new(image.width(), image.height(), image.into_raw())
    }

    /// Image width in pixels.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Interleaved row-major RGB8 pixels.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Returns one RGB8 pixel.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        debug_assert!(x < self.width && y < self.height);
        let index = (y as usize * self.width as usize + x as usize) * 3;
        [
            self.pixels[index],
            self.pixels[index + 1],
            self.pixels[index + 2],
        ]
    }

    /// Samples a pixel with edge-clamped bilinear interpolation.
    #[cfg(test)]
    pub(crate) fn sample_bilinear(&self, x: f32, y: f32) -> [u8; 3] {
        let max_x = self.width.saturating_sub(1) as f32;
        let max_y = self.height.saturating_sub(1) as f32;
        let x = x.clamp(0.0, max_x);
        let y = y.clamp(0.0, max_y);
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = x - x0 as f32;
        let ty = y - y0 as f32;
        let top_left = self.pixel(x0, y0);
        let top_right = self.pixel(x1, y0);
        let bottom_left = self.pixel(x0, y1);
        let bottom_right = self.pixel(x1, y1);
        let mut pixel = [0u8; 3];
        for channel in 0..3 {
            let top =
                f32::from(top_left[channel]) * (1.0 - tx) + f32::from(top_right[channel]) * tx;
            let bottom = f32::from(bottom_left[channel]) * (1.0 - tx)
                + f32::from(bottom_right[channel]) * tx;
            pixel[channel] = (top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0) as u8;
        }
        pixel
    }
}

/// Builds an RGB image filled with one colour for tests and examples.
#[cfg(test)]
pub(crate) fn solid(width: u32, height: u32, pixel: [u8; 3]) -> RgbImage {
    let count = width as usize * height as usize;
    let mut pixels = Vec::with_capacity(count * 3);
    for _ in 0..count {
        pixels.extend_from_slice(&pixel);
    }
    RgbImage::new(width, height, pixels).expect("test image dimensions are valid")
}

/// Builds an RGB image by evaluating each pixel for tests and examples.
#[cfg(test)]
pub(crate) fn from_fn(
    width: u32,
    height: u32,
    mut function: impl FnMut(u32, u32) -> [u8; 3],
) -> RgbImage {
    assert!(
        width > 0 && height > 0,
        "test image dimensions must be non-zero"
    );
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&function(x, y));
        }
    }
    RgbImage::new(width, height, pixels).expect("test image allocation and dimensions are valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_the_backing_length() {
        assert!(RgbImage::new(2, 1, vec![0; 5]).is_err());
        assert!(RgbImage::new(0, 1, Vec::new()).is_err());
    }

    #[test]
    fn samples_between_pixels() {
        let image = from_fn(2, 1, |x, _| [x as u8 * 100, 0, 0]);
        assert_eq!(image.sample_bilinear(0.5, 0.0), [50, 0, 0]);
    }
}
