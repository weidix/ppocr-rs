//! Image preprocessing shared by the end-to-end CPU OCR pipeline.

use crate::ocr::DetectorTransform;
use anyhow::{Result, bail};
use image::{RgbImage, imageops, imageops::FilterType};

const DETECTOR_LIMIT_SIDE: f64 = 736.0;
const DETECTOR_MAX_SIDE: f64 = 4_000.0;
const RECOGNIZER_HEIGHT: u32 = 48;
const RECOGNIZER_MIN_WIDTH: u32 = 320;
const DETECTOR_MEAN_BGR: [f32; 3] = [0.485, 0.456, 0.406];
const DETECTOR_STD_BGR: [f32; 3] = [0.229, 0.224, 0.225];
const RECOGNIZER_MEAN_BGR: [f32; 3] = [0.5, 0.5, 0.5];
const RECOGNIZER_STD_BGR: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Clone, Debug)]
pub(crate) struct PreparedInput {
    pub(crate) data: Vec<f32>,
    pub(crate) height: usize,
    pub(crate) width: usize,
}

impl PreparedInput {
    pub(crate) fn shape(&self) -> [usize; 4] {
        [1, 3, self.height, self.width]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedDetectorInput {
    pub(crate) input: PreparedInput,
    pub(crate) transform: DetectorTransform,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedRecognizerInput {
    pub(crate) input: PreparedInput,
    pub(crate) content_width: usize,
}

pub(crate) fn prepare_detector(
    image: &RgbImage,
    max_side: Option<u32>,
) -> Result<PreparedDetectorInput> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        bail!("detector image must be non-empty");
    }
    let ratio = match max_side {
        Some(limit) if limit > 0 => (f64::from(limit) / f64::from(width.max(height))).min(1.0),
        Some(_) => bail!("detector maximum side must be positive"),
        None => default_detector_ratio(width, height),
    };
    let target_width = aligned_dimension(f64::from(width) * ratio)?;
    let target_height = aligned_dimension(f64::from(height) * ratio)?;
    let resized = imageops::resize(image, target_width, target_height, FilterType::Triangle);
    Ok(PreparedDetectorInput {
        input: normalized_bgr(
            &resized,
            target_width as usize,
            &DETECTOR_MEAN_BGR,
            &DETECTOR_STD_BGR,
        ),
        transform: DetectorTransform::new(width, height, target_width, target_height)?,
    })
}

pub(crate) fn prepare_recognizer(
    image: &RgbImage,
    max_width: u32,
) -> Result<PreparedRecognizerInput> {
    if image.width() == 0 || image.height() == 0 {
        bail!("recognizer image must be non-empty");
    }
    if max_width < RECOGNIZER_MIN_WIDTH {
        bail!("recognizer maximum width must be at least {RECOGNIZER_MIN_WIDTH} pixels");
    }
    let source_ratio = f64::from(image.width()) / f64::from(image.height());
    let resized_width =
        ((f64::from(RECOGNIZER_HEIGHT) * source_ratio).ceil() as u32).clamp(1, max_width);
    let resized = imageops::resize(
        image,
        resized_width,
        RECOGNIZER_HEIGHT,
        FilterType::Triangle,
    );
    Ok(PreparedRecognizerInput {
        input: normalized_bgr(
            &resized,
            max_width as usize,
            &RECOGNIZER_MEAN_BGR,
            &RECOGNIZER_STD_BGR,
        ),
        content_width: resized_width as usize,
    })
}

fn default_detector_ratio(width: u32, height: u32) -> f64 {
    let min_side = f64::from(width.min(height));
    let mut ratio = if min_side < DETECTOR_LIMIT_SIDE {
        DETECTOR_LIMIT_SIDE / min_side
    } else {
        1.0
    };
    if f64::from(width.max(height)) * ratio > DETECTOR_MAX_SIDE {
        ratio = DETECTOR_MAX_SIDE / f64::from(width.max(height));
    }
    ratio
}

fn aligned_dimension(value: f64) -> Result<u32> {
    if !value.is_finite() || value <= 0.0 {
        bail!("invalid resized image dimension {value}");
    }
    let units = (value / 32.0).round().max(1.0);
    if units > f64::from(u32::MAX / 32) {
        bail!("resized image dimension {value} is too large");
    }
    Ok(units as u32 * 32)
}

fn normalized_bgr(
    image: &RgbImage,
    canvas_width: usize,
    mean: &[f32; 3],
    standard_deviation: &[f32; 3],
) -> PreparedInput {
    let (width, height) = image.dimensions();
    let height = height as usize;
    let width = width as usize;
    let mut data = vec![0.0; 3 * height * canvas_width];
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            let bgr = [pixel[2], pixel[1], pixel[0]];
            for channel in 0..3 {
                let value = f32::from(bgr[channel]) / 255.0;
                data[channel * height * canvas_width + y * canvas_width + x] =
                    (value - mean[channel]) / standard_deviation[channel];
            }
        }
    }
    PreparedInput {
        data,
        height,
        width: canvas_width,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn detector_uses_32_pixel_aligned_inputs() {
        let image = RgbImage::from_pixel(1_920, 1_080, Rgb([0, 0, 0]));
        let prepared = prepare_detector(&image, Some(736)).expect("prepare detector input");
        assert_eq!(prepared.input.shape(), [1, 3, 416, 736]);
        assert!((prepared.transform.map_x_to_source(736.0) - 1_920.0).abs() < 1e-4);
        assert!((prepared.transform.map_y_to_source(416.0) - 1_080.0).abs() < 1e-4);
    }

    #[test]
    fn recognizer_pads_and_tracks_content_width() {
        let image = RgbImage::from_pixel(80, 20, Rgb([127, 127, 127]));
        let prepared = prepare_recognizer(&image, 320).expect("prepare recognizer input");
        assert_eq!(prepared.input.shape(), [1, 3, 48, 320]);
        assert_eq!(prepared.content_width, 192);
        let expected = (127.0 / 255.0 - 0.5) / 0.5;
        assert!((prepared.input.data[0] - expected).abs() < 1e-6);
        assert_eq!(prepared.input.data[prepared.content_width], 0.0);
    }
}
