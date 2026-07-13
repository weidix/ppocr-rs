use crate::ocr::DetectorTransform;
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use image::{ImageReader, RgbImage, imageops, imageops::FilterType};
use serde::Serialize;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const DET_MEAN_BGR: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD_BGR: [f32; 3] = [0.229, 0.224, 0.225];
const REC_MEAN_BGR: [f32; 3] = [0.5, 0.5, 0.5];
const REC_STD_BGR: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Clone)]
pub struct PreparedInput {
    pub data: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

impl PreparedInput {
    pub fn to_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        Tensor::from_slice(&self.data, (1, 3, self.height, self.width), device)
    }

    pub fn shape(&self) -> [usize; 4] {
        [1, 3, self.height, self.width]
    }
}

#[derive(Clone)]
pub struct PreparedDetectorInput {
    pub input: PreparedInput,
    pub transform: DetectorTransform,
}

#[derive(Clone)]
pub struct PreparedRecognizerInput {
    pub input: PreparedInput,
    pub content_width: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub fn load_rgb(path: impl AsRef<Path>) -> Result<RgbImage> {
    let path = path.as_ref();
    Ok(ImageReader::open(path)
        .with_context(|| format!("open image {}", path.display()))?
        .decode()
        .with_context(|| format!("decode image {}", path.display()))?
        .to_rgb8())
}

pub fn prepare_detector(image: &RgbImage, max_side: Option<u32>) -> PreparedInput {
    prepare_detector_with_transform(image, max_side)
        .expect("detector input requires a non-empty image")
        .input
}

pub fn prepare_detector_with_transform(
    image: &RgbImage,
    max_side: Option<u32>,
) -> Result<PreparedDetectorInput> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        bail!("detector image must be non-empty");
    }
    let ratio = match max_side {
        Some(limit) => (f64::from(limit) / f64::from(width.max(height))).min(1.0),
        None => {
            let min_side = width.min(height) as f64;
            let mut ratio = if min_side < 736.0 {
                736.0 / min_side
            } else {
                1.0
            };
            if f64::from(width.max(height)) * ratio > 4000.0 {
                ratio = 4000.0 / f64::from(width.max(height));
            }
            ratio
        }
    };
    let target_height = ((f64::from(height) * ratio / 32.0).round() as u32).max(1) * 32;
    let target_width = ((f64::from(width) * ratio / 32.0).round() as u32).max(1) * 32;
    let resized = imageops::resize(image, target_width, target_height, FilterType::Triangle);
    Ok(PreparedDetectorInput {
        input: normalized_bgr(&resized, target_width as usize, &DET_MEAN_BGR, &DET_STD_BGR),
        transform: DetectorTransform::new(width, height, target_width, target_height)?,
    })
}

pub fn longest_annotation_crop(
    annotations: impl AsRef<Path>,
    image_path: impl AsRef<Path>,
    image: &RgbImage,
) -> Result<Crop> {
    let wanted_name = image_path
        .as_ref()
        .file_name()
        .and_then(|name| name.to_str())
        .context("image path does not have a UTF-8 filename")?;
    let reader = BufReader::new(
        File::open(annotations.as_ref())
            .with_context(|| format!("open annotations {}", annotations.as_ref().display()))?,
    );
    let (image_width, image_height) = image.dimensions();
    let mut best = None;

    for line in reader.lines() {
        let line = line?;
        let record: Value = serde_json::from_str(&line)?;
        let record_name = record["image"]
            .as_str()
            .and_then(|value| Path::new(value).file_name())
            .and_then(|value| value.to_str());
        if record_name != Some(wanted_name) {
            continue;
        }
        for detection in record["detections"].as_array().into_iter().flatten() {
            let Some(points) = detection["polygon"].as_array() else {
                continue;
            };
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for point in points {
                let Some(pair) = point.as_array() else {
                    continue;
                };
                let (Some(x), Some(y)) = (
                    pair.first().and_then(Value::as_f64),
                    pair.get(1).and_then(Value::as_f64),
                ) else {
                    continue;
                };
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
            if !min_x.is_finite() || !min_y.is_finite() {
                continue;
            }
            let x0 = min_x.floor().max(0.0) as u32;
            let y0 = min_y.floor().max(0.0) as u32;
            let x1 = max_x.ceil().min(f64::from(image_width)) as u32;
            let y1 = max_y.ceil().min(f64::from(image_height)) as u32;
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let crop = Crop {
                x: x0,
                y: y0,
                width: x1 - x0,
                height: y1 - y0,
            };
            if best
                .as_ref()
                .map(|current: &Crop| crop.width > current.width)
                .unwrap_or(true)
            {
                best = Some(crop);
            }
        }
        break;
    }
    best.context("no annotation crop found for image")
}

pub fn prepare_recognizer(
    image: &RgbImage,
    crop: Crop,
    max_width: Option<u32>,
) -> Result<PreparedInput> {
    Ok(prepare_recognizer_from_image(&crop_image(image, crop)?, max_width)?.input)
}

pub fn prepare_recognizer_from_image(
    image: &RgbImage,
    max_width: Option<u32>,
) -> Result<PreparedRecognizerInput> {
    if image.width() == 0 || image.height() == 0 {
        bail!("recognizer image must be non-empty");
    }
    if let Some(max_width) = max_width {
        if max_width < 320 {
            bail!("recognizer maximum width must be at least 320 pixels");
        }
    }
    let source_ratio = f64::from(image.width()) / f64::from(image.height());
    let canvas_width = max_width.unwrap_or_else(|| {
        ((48.0 * source_ratio.max(320.0 / 48.0)) as u32)
            .max(320)
            .min(3200)
    });
    let resized_width = ((48.0 * source_ratio).ceil() as u32).min(canvas_width);
    let resized = imageops::resize(image, resized_width, 48, FilterType::Triangle);
    Ok(PreparedRecognizerInput {
        input: normalized_bgr(&resized, canvas_width as usize, &REC_MEAN_BGR, &REC_STD_BGR),
        content_width: resized_width as usize,
    })
}

fn crop_image(image: &RgbImage, crop: Crop) -> Result<RgbImage> {
    if crop.width == 0 || crop.height == 0 {
        bail!("recognizer crop must be non-empty");
    }
    let (image_width, image_height) = image.dimensions();
    if crop.x >= image_width
        || crop.y >= image_height
        || crop.width > image_width - crop.x
        || crop.height > image_height - crop.y
    {
        bail!("recognizer crop is outside the image bounds");
    }
    Ok(imageops::crop_imm(image, crop.x, crop.y, crop.width, crop.height).to_image())
}

fn normalized_bgr(
    image: &RgbImage,
    canvas_width: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
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
                    (value - mean[channel]) / std[channel];
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
    fn detector_max_side_preserves_32_pixel_alignment() {
        let image = RgbImage::from_pixel(1920, 1080, Rgb([0, 0, 0]));
        let prepared = prepare_detector(&image, Some(736));
        assert_eq!(prepared.shape(), [1, 3, 416, 736]);

        let small_image = RgbImage::from_pixel(320, 160, Rgb([0, 0, 0]));
        let small_prepared = prepare_detector(&small_image, Some(736));
        assert_eq!(small_prepared.shape(), [1, 3, 160, 320]);
    }

    #[test]
    fn detector_transform_maps_resized_coordinates_back_to_the_source() {
        let image = RgbImage::from_pixel(1920, 1080, Rgb([0, 0, 0]));
        let prepared =
            prepare_detector_with_transform(&image, Some(736)).expect("prepare detector input");

        assert_eq!(prepared.input.shape(), [1, 3, 416, 736]);
        assert!((prepared.transform.map_x_to_source(736.0) - 1920.0).abs() < 1e-4);
        assert!((prepared.transform.map_y_to_source(416.0) - 1080.0).abs() < 1e-4);
    }

    #[test]
    fn recognizer_uses_half_normalization_and_zero_padding() {
        let image = RgbImage::from_pixel(1, 1, Rgb([127, 127, 127]));
        let prepared = prepare_recognizer(
            &image,
            Crop {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            Some(320),
        )
        .expect("prepare recognizer input");

        assert_eq!(prepared.shape(), [1, 3, 48, 320]);
        let expected = (127.0 / 255.0 - 0.5) / 0.5;
        assert!((prepared.data[0] - expected).abs() < 1e-6);
        assert_eq!(prepared.data[48], 0.0);
    }

    #[test]
    fn recognizer_reports_the_unpadded_content_width() {
        let image = RgbImage::from_pixel(80, 20, Rgb([127, 127, 127]));
        let prepared =
            prepare_recognizer_from_image(&image, Some(320)).expect("prepare recognizer input");

        assert_eq!(prepared.input.shape(), [1, 3, 48, 320]);
        assert_eq!(prepared.content_width, 192);
    }
}
