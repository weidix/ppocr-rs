use anyhow::{Context, Result, bail};
use image::{ImageReader, RgbImage, imageops, imageops::FilterType};
use serde_json::Value;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

const DET_MEAN_BGR: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD_BGR: [f32; 3] = [0.229, 0.224, 0.225];
const REC_MEAN_BGR: [f32; 3] = [0.5, 0.5, 0.5];
const REC_STD_BGR: [f32; 3] = [0.5, 0.5, 0.5];

pub const DETECTOR_SHAPE: [usize; 4] = [1, 3, 416, 736];
pub const RECOGNIZER_SHAPE: [usize; 4] = [1, 3, 48, 320];

#[derive(Clone, Debug)]
pub struct PreparedInput {
    pub data: Vec<f32>,
    height: usize,
    width: usize,
}

impl PreparedInput {
    pub fn shape(&self) -> [usize; 4] {
        [1, 3, self.height, self.width]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorTransform {
    source_width: u32,
    source_height: u32,
    content_width: u32,
    content_height: u32,
}

impl DetectorTransform {
    pub fn source_width(self) -> u32 {
        self.source_width
    }

    pub fn source_height(self) -> u32 {
        self.source_height
    }

    pub fn content_width(self) -> u32 {
        self.content_width
    }

    pub fn content_height(self) -> u32 {
        self.content_height
    }

    pub fn map_x_to_source(self, x: f32) -> f32 {
        (x * self.source_width as f32 / self.content_width as f32)
            .clamp(0.0, self.source_width as f32)
    }

    pub fn map_y_to_source(self, y: f32) -> f32 {
        (y * self.source_height as f32 / self.content_height as f32)
            .clamp(0.0, self.source_height as f32)
    }
}

#[derive(Clone, Debug)]
pub struct FixedDetectorInput {
    pub input: PreparedInput,
    pub transform: DetectorTransform,
}

#[derive(Clone, Debug)]
pub struct FixedRecognizerInput {
    pub input: PreparedInput,
    pub content_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Crop {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

pub fn load_rgb(path: impl AsRef<Path>) -> Result<RgbImage> {
    let path = path.as_ref();
    Ok(ImageReader::open(path)
        .with_context(|| format!("open image {}", path.display()))?
        .decode()
        .with_context(|| format!("decode image {}", path.display()))?
        .to_rgb8())
}

pub fn prepare_fixed_detector(image: &RgbImage) -> Result<PreparedInput> {
    Ok(prepare_fixed_detector_with_transform(image)?.input)
}

pub fn prepare_fixed_detector_with_transform(image: &RgbImage) -> Result<FixedDetectorInput> {
    let (source_width, source_height) = image.dimensions();
    if source_width == 0 || source_height == 0 {
        bail!("detector image must be non-empty");
    }

    let target_width = DETECTOR_SHAPE[3] as u32;
    let target_height = DETECTOR_SHAPE[2] as u32;
    let scale = (target_width as f64 / source_width as f64)
        .min(target_height as f64 / source_height as f64);
    let content_width = ((source_width as f64 * scale).round() as u32).clamp(1, target_width);
    let content_height = ((source_height as f64 * scale).round() as u32).clamp(1, target_height);
    let resized = imageops::resize(image, content_width, content_height, FilterType::Triangle);
    let input = normalized_bgr(
        &resized,
        target_height as usize,
        target_width as usize,
        &DET_MEAN_BGR,
        &DET_STD_BGR,
    );

    Ok(FixedDetectorInput {
        input,
        transform: DetectorTransform {
            source_width,
            source_height,
            content_width,
            content_height,
        },
    })
}

pub fn prepare_fixed_recognizer(
    annotations: Option<&Path>,
    image_path: &Path,
    image: &RgbImage,
) -> Result<PreparedInput> {
    let crop = match annotations {
        Some(annotations) => longest_annotation_crop(annotations, image_path, image)?,
        None => Crop {
            x: 0,
            y: 0,
            width: image.width(),
            height: image.height(),
        },
    };
    Ok(prepare_fixed_recognizer_from_crop(image, crop)?.input)
}

pub fn prepare_fixed_recognizer_from_image(image: &RgbImage) -> Result<FixedRecognizerInput> {
    if image.width() == 0 || image.height() == 0 {
        bail!("recognizer image must be non-empty");
    }
    prepare_recognizer_image(image, RECOGNIZER_SHAPE[3] as u32)
}

fn prepare_fixed_recognizer_from_crop(
    image: &RgbImage,
    crop: Crop,
) -> Result<FixedRecognizerInput> {
    let cropped = crop_image(image, crop)?;
    prepare_fixed_recognizer_from_image(&cropped)
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

pub fn prepare_detector(image: &RgbImage, max_side: Option<u32>) -> PreparedInput {
    let (width, height) = image.dimensions();
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
    normalized_bgr(
        &resized,
        target_height as usize,
        target_width as usize,
        &DET_MEAN_BGR,
        &DET_STD_BGR,
    )
}

pub fn prepare_recognizer(
    image: &RgbImage,
    crop: Crop,
    max_width: Option<u32>,
) -> Result<PreparedInput> {
    let cropped = crop_image(image, crop)?;
    let canvas_width = recognizer_canvas_width(&cropped, max_width)?;
    Ok(prepare_recognizer_image(&cropped, canvas_width)?.input)
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

fn recognizer_canvas_width(image: &RgbImage, max_width: Option<u32>) -> Result<u32> {
    if image.width() == 0 || image.height() == 0 {
        bail!("recognizer image must be non-empty");
    }
    if max_width == Some(0) {
        bail!("recognizer maximum width must be greater than zero");
    }
    let source_ratio = f64::from(image.width()) / f64::from(image.height());
    Ok(max_width.unwrap_or_else(|| {
        ((48.0 * source_ratio.max(320.0 / 48.0)) as u32)
            .max(320)
            .min(3200)
    }))
}

fn prepare_recognizer_image(image: &RgbImage, canvas_width: u32) -> Result<FixedRecognizerInput> {
    if canvas_width == 0 {
        bail!("recognizer canvas width must be greater than zero");
    }
    let source_ratio = f64::from(image.width()) / f64::from(image.height());
    let content_width = ((48.0 * source_ratio).ceil() as u32).clamp(1, canvas_width);
    let resized = imageops::resize(image, content_width, 48, FilterType::Triangle);
    let input = normalized_bgr(
        &resized,
        RECOGNIZER_SHAPE[2],
        canvas_width as usize,
        &REC_MEAN_BGR,
        &REC_STD_BGR,
    );
    Ok(FixedRecognizerInput {
        input,
        content_width: content_width as usize,
    })
}

fn normalized_bgr(
    image: &RgbImage,
    canvas_height: usize,
    canvas_width: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> PreparedInput {
    let (width, height) = image.dimensions();
    let height = height as usize;
    let width = width as usize;
    debug_assert!(height <= canvas_height);
    debug_assert!(width <= canvas_width);
    let mut data = vec![0.0; 3 * canvas_height * canvas_width];
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            let bgr = [pixel[2], pixel[1], pixel[0]];
            for channel in 0..3 {
                let value = f32::from(bgr[channel]) / 255.0;
                data[channel * canvas_height * canvas_width + y * canvas_width + x] =
                    (value - mean[channel]) / std[channel];
            }
        }
    }
    PreparedInput {
        data,
        height: canvas_height,
        width: canvas_width,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn fixed_detector_letterboxes_arbitrary_aspect_ratios() {
        let image = RgbImage::from_pixel(400, 400, Rgb([0, 0, 0]));
        let prepared = prepare_fixed_detector_with_transform(&image).expect("prepare detector");

        assert_eq!(prepared.input.shape(), DETECTOR_SHAPE);
        assert_eq!(prepared.transform.content_width(), 416);
        assert_eq!(prepared.transform.content_height(), 416);
        assert!((prepared.input.data[0] + 0.485 / 0.229).abs() < 1e-6);
        assert_eq!(prepared.input.data[416], 0.0);
        assert!((prepared.transform.map_x_to_source(416.0) - 400.0).abs() < 1e-6);
    }

    #[test]
    fn fixed_detector_shape_and_bgr_normalization_match_the_parent_path() {
        let image = RgbImage::from_pixel(1920, 1080, Rgb([0, 0, 0]));
        let prepared = prepare_fixed_detector(&image).expect("prepare detector");
        assert_eq!(prepared.shape(), DETECTOR_SHAPE);
        assert!((prepared.data[0] + 0.485 / 0.229).abs() < 1e-6);
    }

    #[test]
    fn recognizer_uses_half_normalization_and_zero_padding() {
        let image = RgbImage::from_pixel(1, 1, Rgb([127, 127, 127]));
        let prepared = prepare_fixed_recognizer_from_image(&image).expect("prepare recognizer");

        assert_eq!(prepared.input.shape(), RECOGNIZER_SHAPE);
        assert_eq!(prepared.content_width, 48);
        let expected = (127.0 / 255.0 - 0.5) / 0.5;
        assert!((prepared.input.data[0] - expected).abs() < 1e-6);
        assert_eq!(prepared.input.data[48], 0.0);
    }

    #[test]
    fn recognizer_rejects_out_of_bounds_crops() {
        let image = RgbImage::from_pixel(2, 2, Rgb([0, 0, 0]));
        assert!(
            prepare_recognizer(
                &image,
                Crop {
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 2,
                },
                Some(320),
            )
            .is_err()
        );
    }
}
