use anyhow::{Context, Result, bail};
use image::{Rgb, RgbImage, imageops};
use serde::Serialize;
use std::{collections::VecDeque, fs, path::Path};

#[cfg(feature = "cpu")]
use crate::{
    cpu::{CpuOptions, Detector, ModelSize, Recognizer, Tensor},
    models::ModelStore,
    preprocess::{prepare_detector, prepare_recognizer},
};
#[cfg(feature = "cpu")]
use anyhow::ensure;
#[cfg(feature = "cpu")]
use image::ImageReader;

pub const RECOGNIZER_INPUT_HEIGHT: usize = 48;
pub const RECOGNIZER_INPUT_WIDTH: usize = 320;

const MAX_RECTIFIED_EDGE: f32 = 4_096.0;
const MAX_RECTIFIED_PIXELS: f32 = 8_000_000.0;
const MAX_RECOGNITION_CHUNKS: usize = 64;
const PROBABILITY_EPSILON: f32 = 1e-8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorTransform {
    source_width: u32,
    source_height: u32,
    content_width: u32,
    content_height: u32,
}

impl DetectorTransform {
    pub fn new(
        source_width: u32,
        source_height: u32,
        content_width: u32,
        content_height: u32,
    ) -> Result<Self> {
        if source_width == 0 || source_height == 0 || content_width == 0 || content_height == 0 {
            bail!("detector transform dimensions must be non-zero");
        }
        Ok(Self {
            source_width,
            source_height,
            content_width,
            content_height,
        })
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Point(pub f32, pub f32);

#[derive(Clone, Debug, Serialize)]
pub struct Detection {
    pub polygon: [Point; 4],
    pub score: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DetectorPostprocessOptions {
    pub binary_threshold: f32,
    pub box_threshold: f32,
    pub min_area: usize,
    pub unclip_ratio: f32,
    pub max_boxes: usize,
}

impl Default for DetectorPostprocessOptions {
    fn default() -> Self {
        Self {
            binary_threshold: 0.2,
            box_threshold: 0.4,
            min_area: 3,
            unclip_ratio: 1.4,
            max_boxes: 1_000,
        }
    }
}

impl DetectorPostprocessOptions {
    pub fn validate(self) -> Result<()> {
        validate_probability(self.binary_threshold, "detector binary threshold")?;
        validate_probability(self.box_threshold, "detector box threshold")?;
        if self.min_area == 0 {
            bail!("detector minimum area must be at least one pixel");
        }
        if self.max_boxes == 0 {
            bail!("detector maximum box count must be at least one");
        }
        if !self.unclip_ratio.is_finite() || self.unclip_ratio <= 0.0 {
            bail!("detector unclip ratio must be a finite value greater than zero");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DecodedText {
    pub text: String,
    pub score: f32,
}

/// Configuration for the end-to-end CPU OCR pipeline.
#[cfg(feature = "cpu")]
#[derive(Clone, Debug)]
pub struct OcrOptions {
    /// Detector model tier.
    pub detector_size: ModelSize,
    /// Recognizer model tier.
    pub recognizer_size: ModelSize,
    /// CPU runtime settings shared by the detector and recognizer.
    pub cpu: CpuOptions,
    /// Detector output postprocessing settings.
    pub detector_postprocess: DetectorPostprocessOptions,
    /// Optional maximum detector image side. `None` uses the released model's
    /// default resize policy.
    pub detector_max_side: Option<u32>,
    /// Width of each recognizer canvas and the maximum width of one text crop.
    pub recognizer_max_width: u32,
}

#[cfg(feature = "cpu")]
impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            detector_size: ModelSize::Tiny,
            recognizer_size: ModelSize::Tiny,
            cpu: CpuOptions::default(),
            detector_postprocess: DetectorPostprocessOptions::default(),
            detector_max_side: None,
            recognizer_max_width: RECOGNIZER_INPUT_WIDTH as u32,
        }
    }
}

#[cfg(feature = "cpu")]
impl OcrOptions {
    /// Validates options before model resolution or inference.
    pub fn validate(&self) -> Result<()> {
        self.detector_postprocess.validate()?;
        ensure!(self.cpu.threads > 0, "CPU thread count must be positive");
        ensure!(
            self.recognizer_max_width >= RECOGNIZER_INPUT_WIDTH as u32,
            "recognizer maximum width must be at least {RECOGNIZER_INPUT_WIDTH}"
        );
        if let Some(max_side) = self.detector_max_side {
            ensure!(max_side > 0, "detector maximum side must be positive");
        }
        Ok(())
    }
}

/// One recognized text region.
#[cfg(feature = "cpu")]
#[derive(Clone, Debug, Serialize)]
pub struct OcrLine {
    /// Text-region quadrilateral in source-image coordinates.
    pub polygon: [Point; 4],
    /// Confidence emitted by the detector.
    pub detection_score: f32,
    /// CTC-decoded text.
    pub text: String,
    /// Geometric mean confidence of emitted recognition tokens.
    pub recognition_score: f32,
}

/// The complete result of OCR on one image.
#[cfg(feature = "cpu")]
#[derive(Clone, Debug, Serialize)]
pub struct OcrResult {
    /// Source image dimensions as `[width, height]`.
    pub source_size: [u32; 2],
    /// Detector input dimensions as `[width, height]` after resizing.
    pub detector_input_size: [usize; 2],
    /// Text regions in natural reading order.
    pub lines: Vec<OcrLine>,
}

#[cfg(feature = "cpu")]
impl OcrResult {
    /// Returns recognized non-empty lines joined with newlines.
    pub fn text(&self) -> String {
        self.lines
            .iter()
            .filter(|line| !line.text.is_empty())
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// An end-to-end PP-OCRv6 CPU inference engine.
#[cfg(feature = "cpu")]
pub struct OcrEngine {
    detector: Detector,
    recognizer: Recognizer,
    dictionary: Vec<String>,
    options: OcrOptions,
}

#[cfg(feature = "cpu")]
impl OcrEngine {
    /// Downloads missing pinned models through `store`, then loads an OCR engine.
    pub fn load_from_store(store: &ModelStore, options: OcrOptions) -> Result<Self> {
        let paths = store.ensure_pair(options.detector_size, options.recognizer_size)?;
        Self::load(
            &paths.detector.weights,
            &paths.recognizer.weights,
            &paths.recognizer.inference,
            options,
        )
    }

    /// Loads an engine from explicit detector, recognizer, and dictionary paths.
    pub fn load(
        detector_model: impl AsRef<Path>,
        recognizer_model: impl AsRef<Path>,
        dictionary: impl AsRef<Path>,
        options: OcrOptions,
    ) -> Result<Self> {
        options.validate()?;
        let detector_model = detector_model.as_ref();
        let recognizer_model = recognizer_model.as_ref();
        let dictionary_path = dictionary.as_ref();
        let dictionary = load_dictionary(dictionary_path)?;
        validate_recognizer_dictionary(&dictionary, options.recognizer_size)?;
        let detector = Detector::load(detector_model, options.detector_size, options.cpu)
            .with_context(|| format!("load detector {}", detector_model.display()))?;
        let recognizer =
            Recognizer::load(recognizer_model, options.recognizer_size, options.cpu)
                .with_context(|| format!("load recognizer {}", recognizer_model.display()))?;
        Ok(Self {
            detector,
            recognizer,
            dictionary,
            options,
        })
    }

    /// Decodes an image file into text regions.
    pub fn recognize_path(&self, path: impl AsRef<Path>) -> Result<OcrResult> {
        let path = path.as_ref();
        let image = ImageReader::open(path)
            .with_context(|| format!("open image {}", path.display()))?
            .decode()
            .with_context(|| format!("decode image {}", path.display()))?
            .to_rgb8();
        self.recognize(&image)
    }

    /// Decodes an RGB image into text regions.
    pub fn recognize(&self, image: &RgbImage) -> Result<OcrResult> {
        let source_size = [image.width(), image.height()];
        let prepared = prepare_detector(image, self.options.detector_max_side)?;
        let detector_input_size = [prepared.input.width, prepared.input.height];
        let detector_input = Tensor::from_f32(prepared.input.shape(), prepared.input.data)?;
        let detector_output = self.detector.run(detector_input)?;
        let detections = extract_detections(
            detector_output.as_f32()?,
            detector_output.shape(),
            prepared.transform,
            self.options.detector_postprocess,
        )?;

        let mut lines = Vec::with_capacity(detections.len());
        for detection in detections {
            let crop = rectify_text_crop(image, detection.polygon)?;
            let mut decoded_chunks = Vec::new();
            for chunk in split_recognition_crop_for_input(
                &crop,
                RECOGNIZER_INPUT_HEIGHT,
                self.options.recognizer_max_width as usize,
            )? {
                let prepared = prepare_recognizer(&chunk, self.options.recognizer_max_width)?;
                let content_width = prepared.content_width;
                let input_width = prepared.input.width;
                let recognizer_input =
                    Tensor::from_f32(prepared.input.shape(), prepared.input.data)?;
                let recognizer_output = self.recognizer.run(recognizer_input)?;
                decoded_chunks.push(decode_ctc_greedy_for_input(
                    recognizer_output.as_f32()?,
                    recognizer_output.shape(),
                    &self.dictionary,
                    content_width,
                    input_width,
                )?);
            }
            let decoded = join_decoded_texts(&decoded_chunks);
            lines.push(OcrLine {
                polygon: detection.polygon,
                detection_score: detection.score,
                text: decoded.text,
                recognition_score: decoded.score,
            });
        }

        Ok(OcrResult {
            source_size,
            detector_input_size,
            lines,
        })
    }
}

#[cfg(feature = "cpu")]
fn validate_recognizer_dictionary(dictionary: &[String], size: ModelSize) -> Result<()> {
    let classes = size.recognizer_classes();
    ensure!(
        classes == dictionary.len() + 1 || classes == dictionary.len() + 2,
        "{} recognizer has {classes} output classes, but the dictionary has {} entries",
        size.as_str(),
        dictionary.len()
    );
    Ok(())
}

pub fn split_recognition_crop_for_input(
    image: &RgbImage,
    input_height: usize,
    input_width: usize,
) -> Result<Vec<RgbImage>> {
    if image.width() == 0 || image.height() == 0 {
        bail!("cannot split an empty recognition crop");
    }
    if input_height == 0 || input_width == 0 {
        bail!("recognizer input dimensions must be non-zero");
    }
    let max_width = (image.height() as usize)
        .checked_mul(input_width)
        .context("recognition crop width overflow")?
        / input_height;
    let max_width = u32::try_from(max_width.max(1)).context("recognition crop width overflow")?;
    if image.width() <= max_width {
        return Ok(vec![image.clone()]);
    }

    let overlap = if max_width > 1 {
        (max_width / 3).clamp(1, max_width - 1)
    } else {
        0
    };
    let step = max_width.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0u32;
    loop {
        if chunks.len() == MAX_RECOGNITION_CHUNKS {
            bail!(
                "recognition crop requires more than {MAX_RECOGNITION_CHUNKS} fixed-width chunks"
            );
        }
        let width = (image.width() - start).min(max_width);
        chunks.push(imageops::crop_imm(image, start, 0, width, image.height()).to_image());
        if start + width >= image.width() {
            break;
        }
        start += step;
    }
    Ok(chunks)
}

pub fn join_decoded_texts(parts: &[DecodedText]) -> DecodedText {
    let mut text = String::new();
    let mut score_sum = 0.0;
    let mut score_weight = 0usize;
    for part in parts {
        let added = append_with_overlap(&mut text, &part.text);
        if added != 0 {
            score_sum += f64::from(part.score) * added as f64;
            score_weight += added;
        }
    }
    DecodedText {
        text,
        score: if score_weight == 0 {
            0.0
        } else {
            (score_sum / score_weight as f64) as f32
        },
    }
}

pub fn extract_detections(
    values: &[f32],
    shape: &[usize],
    transform: DetectorTransform,
    options: DetectorPostprocessOptions,
) -> Result<Vec<Detection>> {
    options.validate()?;
    let (height, width) = detector_output_shape(shape, values.len())?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("detector output contains non-finite values");
    }

    let content_width = usize::min(transform.content_width() as usize, width);
    let content_height = usize::min(transform.content_height() as usize, height);
    let visited_len = content_width
        .checked_mul(content_height)
        .context("detector content area overflow")?;
    let mut visited = vec![false; visited_len];
    let mut detections = Vec::new();
    for y in 0..content_height {
        for x in 0..content_width {
            let active_index = y * content_width + x;
            if visited[active_index] || values[y * width + x] < options.binary_threshold {
                continue;
            }
            let component = collect_component(
                values,
                width,
                content_width,
                content_height,
                x,
                y,
                options.binary_threshold,
                &mut visited,
            );
            if component.points.len() < options.min_area {
                continue;
            }
            let score = (component.score_sum / component.points.len() as f64) as f32;
            if score < options.box_threshold {
                continue;
            }
            let polygon = fit_rotated_box(&component.points, options.unclip_ratio).map(|point| {
                Point(
                    transform.map_x_to_source(point.0),
                    transform.map_y_to_source(point.1),
                )
            });
            detections.push(Detection { polygon, score });
        }
    }
    sort_detections(&mut detections);
    if detections.len() > options.max_boxes {
        detections.sort_by(|left, right| {
            right.score.total_cmp(&left.score).then_with(|| {
                let left_center = polygon_center(left.polygon);
                let right_center = polygon_center(right.polygon);
                left_center
                    .1
                    .total_cmp(&right_center.1)
                    .then_with(|| left_center.0.total_cmp(&right_center.0))
            })
        });
        detections.truncate(options.max_boxes);
        sort_detections(&mut detections);
    }
    Ok(detections)
}

pub fn rectify_text_crop(image: &RgbImage, polygon: [Point; 4]) -> Result<RgbImage> {
    if image.width() == 0 || image.height() == 0 {
        bail!("cannot rectify a crop from an empty image");
    }
    if polygon
        .iter()
        .any(|point| !point.0.is_finite() || !point.1.is_finite())
    {
        bail!("text polygon contains non-finite coordinates");
    }

    let width = distance(polygon[0], polygon[1]).max(distance(polygon[3], polygon[2]));
    let height = distance(polygon[0], polygon[3]).max(distance(polygon[1], polygon[2]));
    let (output_width, output_height) = bounded_crop_dimensions(width, height)?;
    let mut output = RgbImage::new(output_width, output_height);
    for y in 0..output_height {
        let v = (y as f32 + 0.5) / output_height as f32;
        for x in 0..output_width {
            let u = (x as f32 + 0.5) / output_width as f32;
            let source = bilinear_quad(polygon, u, v);
            output.put_pixel(
                x,
                y,
                sample_bilinear(image, Point(source.0 - 0.5, source.1 - 0.5)),
            );
        }
    }
    Ok(output)
}

pub fn load_dictionary(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .with_context(|| format!("read recognition dictionary {}", path.display()))?;
    parse_dictionary(&contents)
        .with_context(|| format!("parse recognition dictionary {}", path.display()))
}

pub fn decode_ctc_greedy_for_input(
    values: &[f32],
    shape: &[usize],
    dictionary: &[String],
    content_width: usize,
    input_width: usize,
) -> Result<DecodedText> {
    let (time_steps, classes) = recognizer_output_shape(shape, values.len())?;
    let implicit_space_class = classes == dictionary.len().saturating_add(2);
    if classes != dictionary.len() + 1 && !implicit_space_class {
        bail!(
            "recognizer output has {classes} classes, but the dictionary has {} entries; expected dictionary entries plus the CTC blank class",
            dictionary.len()
        );
    }
    if content_width == 0 || content_width > input_width {
        bail!("recognizer content width {content_width} is outside 1..={input_width}");
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("recognizer output contains non-finite values");
    }

    let valid_steps = content_width
        .checked_mul(time_steps)
        .context("recognizer time-step calculation overflow")?
        .div_ceil(input_width);
    let valid_steps = valid_steps.clamp(1, time_steps);
    let mut text = String::new();
    let mut log_score = 0.0f64;
    let mut emitted = 0usize;
    let mut previous = 0usize;
    for time in 0..valid_steps {
        let row = &values[time * classes..(time + 1) * classes];
        let (class, value) = argmax(row);
        if class != 0 && class != previous {
            if implicit_space_class && class == classes - 1 {
                text.push(' ');
            } else {
                text.push_str(&dictionary[class - 1]);
            }
            log_score += f64::from(row_probability(row, value).max(PROBABILITY_EPSILON)).ln();
            emitted += 1;
        }
        previous = class;
    }
    Ok(DecodedText {
        text,
        score: if emitted == 0 {
            0.0
        } else {
            (log_score / emitted as f64).exp() as f32
        },
    })
}

fn parse_dictionary(contents: &str) -> Result<Vec<String>> {
    let mut entries = parse_paddlex_character_dict(contents)?.unwrap_or_else(|| {
        contents
            .lines()
            .map(|line| line.strip_suffix('\r').unwrap_or(line).to_owned())
            .collect()
    });
    if let Some(first) = entries.first_mut() {
        *first = first.trim_start_matches('\u{feff}').to_owned();
    }
    if entries.is_empty() {
        bail!("recognition dictionary is empty");
    }
    if let Some((line, _)) = entries
        .iter()
        .enumerate()
        .find(|(_, entry)| entry.is_empty())
    {
        bail!(
            "recognition dictionary contains an empty entry on line {}",
            line + 1
        );
    }
    Ok(entries)
}

fn parse_paddlex_character_dict(contents: &str) -> Result<Option<Vec<String>>> {
    let lines = contents.lines().collect::<Vec<_>>();
    for (index, line) in lines.iter().enumerate() {
        if line.trim() != "character_dict:" {
            continue;
        }
        let indentation = line.len() - line.trim_start().len();
        let mut entries = Vec::new();
        for (offset, entry_line) in lines[index + 1..].iter().enumerate() {
            let entry_line = entry_line.strip_suffix('\r').unwrap_or(entry_line);
            if entry_line.trim().is_empty() {
                continue;
            }
            let trimmed = entry_line.trim_start();
            let Some(value) = trimmed.strip_prefix("- ") else {
                if entry_line.len() - trimmed.len() <= indentation {
                    break;
                }
                bail!(
                    "expected a character list entry on line {}",
                    index + offset + 2
                );
            };
            entries.push(parse_yaml_scalar(value, index + offset + 2)?);
        }
        return Ok(Some(entries));
    }
    Ok(None)
}

fn parse_yaml_scalar(value: &str, line: usize) -> Result<String> {
    if value.starts_with('\'') {
        if value.len() < 2 || !value.ends_with('\'') {
            bail!("unterminated single-quoted character on line {line}");
        }
        return Ok(value[1..value.len() - 1].replace("''", "'"));
    }
    if value.starts_with('"') {
        if value.len() < 2 || !value.ends_with('"') {
            bail!("unterminated double-quoted character on line {line}");
        }
        return Ok(value[1..value.len() - 1].to_owned());
    }
    Ok(value.to_owned())
}

fn detector_output_shape(shape: &[usize], value_len: usize) -> Result<(usize, usize)> {
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] == 0 || shape[3] == 0 {
        bail!("detector output shape {shape:?}, expected [1, 1, height, width]");
    }
    let expected = shape[2]
        .checked_mul(shape[3])
        .context("detector output shape overflow")?;
    if value_len != expected {
        bail!("detector output has {value_len} values, expected {expected} for shape {shape:?}");
    }
    Ok((shape[2], shape[3]))
}

fn recognizer_output_shape(shape: &[usize], value_len: usize) -> Result<(usize, usize)> {
    if shape.len() != 3 || shape[0] != 1 || shape[1] == 0 || shape[2] < 2 {
        bail!("recognizer output shape {shape:?}, expected [1, time, classes>=2]");
    }
    let expected = shape[1]
        .checked_mul(shape[2])
        .context("recognizer output shape overflow")?;
    if value_len != expected {
        bail!("recognizer output has {value_len} values, expected {expected} for shape {shape:?}");
    }
    Ok((shape[1], shape[2]))
}

fn validate_probability(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("{name} must be a finite value between zero and one");
    }
    Ok(())
}

struct Component {
    points: Vec<Point>,
    score_sum: f64,
}

#[allow(clippy::too_many_arguments)]
fn collect_component(
    values: &[f32],
    output_width: usize,
    content_width: usize,
    content_height: usize,
    start_x: usize,
    start_y: usize,
    threshold: f32,
    visited: &mut [bool],
) -> Component {
    const NEIGHBORS: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    let mut queue = VecDeque::new();
    queue.push_back((start_x, start_y));
    visited[start_y * content_width + start_x] = true;
    let mut points = Vec::new();
    let mut score_sum = 0.0;
    while let Some((x, y)) = queue.pop_front() {
        points.push(Point(x as f32 + 0.5, y as f32 + 0.5));
        score_sum += f64::from(values[y * output_width + x]);
        for (offset_x, offset_y) in NEIGHBORS {
            let next_x = x as isize + offset_x;
            let next_y = y as isize + offset_y;
            if next_x < 0
                || next_y < 0
                || next_x >= content_width as isize
                || next_y >= content_height as isize
            {
                continue;
            }
            let next_x = next_x as usize;
            let next_y = next_y as usize;
            let next_index = next_y * content_width + next_x;
            if !visited[next_index] && values[next_y * output_width + next_x] >= threshold {
                visited[next_index] = true;
                queue.push_back((next_x, next_y));
            }
        }
    }
    Component { points, score_sum }
}

fn fit_rotated_box(points: &[Point], unclip_ratio: f32) -> [Point; 4] {
    let count = points.len() as f32;
    let center = Point(
        points.iter().map(|point| point.0).sum::<f32>() / count,
        points.iter().map(|point| point.1).sum::<f32>() / count,
    );
    let (cov_xx, cov_xy, cov_yy) = points.iter().fold((0.0, 0.0, 0.0), |acc, point| {
        let dx = point.0 - center.0;
        let dy = point.1 - center.1;
        (acc.0 + dx * dx, acc.1 + dx * dy, acc.2 + dy * dy)
    });
    let angle = 0.5 * (2.0 * cov_xy).atan2(cov_xx - cov_yy);
    let mut axis = Point(angle.cos(), angle.sin());
    if axis.0 < 0.0 || (axis.0.abs() < f32::EPSILON && axis.1 < 0.0) {
        axis = Point(-axis.0, -axis.1);
    }
    let normal = Point(-axis.1, axis.0);
    let (min_axis, max_axis, min_normal, max_normal) = points.iter().fold(
        (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ),
        |(min_axis, max_axis, min_normal, max_normal), point| {
            let delta = Point(point.0 - center.0, point.1 - center.1);
            let along_axis = dot(delta, axis);
            let along_normal = dot(delta, normal);
            (
                min_axis.min(along_axis),
                max_axis.max(along_axis),
                min_normal.min(along_normal),
                max_normal.max(along_normal),
            )
        },
    );
    let axis_center = (min_axis + max_axis) * 0.5;
    let normal_center = (min_normal + max_normal) * 0.5;
    let half_axis = ((max_axis - min_axis + 1.0) * unclip_ratio) * 0.5;
    let half_normal = ((max_normal - min_normal + 1.0) * unclip_ratio) * 0.5;
    let center = add(
        center,
        add(scale(axis, axis_center), scale(normal, normal_center)),
    );
    [
        add(
            center,
            add(scale(axis, -half_axis), scale(normal, -half_normal)),
        ),
        add(
            center,
            add(scale(axis, half_axis), scale(normal, -half_normal)),
        ),
        add(
            center,
            add(scale(axis, half_axis), scale(normal, half_normal)),
        ),
        add(
            center,
            add(scale(axis, -half_axis), scale(normal, half_normal)),
        ),
    ]
}

fn sort_detections(detections: &mut [Detection]) {
    detections.sort_by(|left, right| {
        let left_center = polygon_center(left.polygon);
        let right_center = polygon_center(right.polygon);
        left_center
            .1
            .total_cmp(&right_center.1)
            .then_with(|| left_center.0.total_cmp(&right_center.0))
            .then_with(|| left.score.total_cmp(&right.score))
    });
}

fn polygon_center(polygon: [Point; 4]) -> Point {
    Point(
        polygon.iter().map(|point| point.0).sum::<f32>() / polygon.len() as f32,
        polygon.iter().map(|point| point.1).sum::<f32>() / polygon.len() as f32,
    )
}

fn bounded_crop_dimensions(width: f32, height: f32) -> Result<(u32, u32)> {
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        bail!("text polygon has an invalid crop size {width} by {height}");
    }
    let edge_scale = (MAX_RECTIFIED_EDGE / width.max(height)).min(1.0);
    let pixel_scale = (MAX_RECTIFIED_PIXELS / (width * height)).sqrt().min(1.0);
    let scale = edge_scale.min(pixel_scale);
    Ok((
        (width * scale).round().clamp(1.0, MAX_RECTIFIED_EDGE) as u32,
        (height * scale).round().clamp(1.0, MAX_RECTIFIED_EDGE) as u32,
    ))
}

fn bilinear_quad(polygon: [Point; 4], u: f32, v: f32) -> Point {
    let top = add(scale(polygon[0], 1.0 - u), scale(polygon[1], u));
    let bottom = add(scale(polygon[3], 1.0 - u), scale(polygon[2], u));
    add(scale(top, 1.0 - v), scale(bottom, v))
}

fn sample_bilinear(image: &RgbImage, point: Point) -> Rgb<u8> {
    let max_x = image.width().saturating_sub(1) as f32;
    let max_y = image.height().saturating_sub(1) as f32;
    let x = point.0.clamp(0.0, max_x);
    let y = point.1.clamp(0.0, max_y);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let top_left = image.get_pixel(x0, y0).0;
    let top_right = image.get_pixel(x1, y0).0;
    let bottom_left = image.get_pixel(x0, y1).0;
    let bottom_right = image.get_pixel(x1, y1).0;
    let mut pixel = [0u8; 3];
    for channel in 0..3 {
        let top = f32::from(top_left[channel]) * (1.0 - tx) + f32::from(top_right[channel]) * tx;
        let bottom =
            f32::from(bottom_left[channel]) * (1.0 - tx) + f32::from(bottom_right[channel]) * tx;
        pixel[channel] = (top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0) as u8;
    }
    Rgb(pixel)
}

fn row_probability(row: &[f32], value: f32) -> f32 {
    let sum = row.iter().sum::<f32>();
    if row.iter().all(|candidate| *candidate >= 0.0) && (sum - 1.0).abs() <= 0.01 {
        return value.clamp(0.0, 1.0);
    }
    let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator = row
        .iter()
        .map(|candidate| (*candidate - maximum).exp())
        .sum::<f32>();
    ((value - maximum).exp() / denominator).clamp(0.0, 1.0)
}

fn argmax(row: &[f32]) -> (usize, f32) {
    row.iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("recognizer output rows are non-empty")
}

fn append_with_overlap(target: &mut String, next: &str) -> usize {
    if next.is_empty() {
        return 0;
    }
    if target.is_empty() {
        target.push_str(next);
        return next.chars().count();
    }
    let target_chars = target.chars().collect::<Vec<_>>();
    let next_chars = next.chars().collect::<Vec<_>>();
    let maximum = target_chars.len().min(next_chars.len()).min(32);
    let overlap = (1..=maximum)
        .rev()
        .find(|size| target_chars[target_chars.len() - *size..] == next_chars[..*size])
        .unwrap_or(0);
    target.extend(next_chars[overlap..].iter().copied());
    next_chars.len() - overlap
}

fn dot(left: Point, right: Point) -> f32 {
    left.0 * right.0 + left.1 * right.1
}

fn add(left: Point, right: Point) -> Point {
    Point(left.0 + right.0, left.1 + right.1)
}

fn scale(point: Point, factor: f32) -> Point {
    Point(point.0 * factor, point.1 * factor)
}

fn distance(left: Point, right: Point) -> f32 {
    ((left.0 - right.0).powi(2) + (left.1 - right.1).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn detector_postprocess_maps_components_to_source_coordinates() {
        let mut values = vec![0.0; 416 * 736];
        for y in 20..22 {
            for x in 10..14 {
                values[y * 736 + x] = 0.9;
            }
        }
        let detections = extract_detections(
            &values,
            &[1, 1, 416, 736],
            DetectorTransform::new(736, 416, 736, 416).expect("transform"),
            DetectorPostprocessOptions {
                min_area: 4,
                unclip_ratio: 1.0,
                ..Default::default()
            },
        )
        .expect("postprocess detector");

        assert_eq!(detections.len(), 1);
        assert!((detections[0].score - 0.9).abs() < 1e-6);
        assert!((detections[0].polygon[0].0 - 10.0).abs() < 1e-4);
    }

    #[test]
    fn dynamic_ctc_ignores_padded_time_steps() {
        let dictionary = vec!["A".to_owned()];
        let values = [
            0.1, 0.9, // A
            0.1, 0.9, // collapsed A
            0.1, 0.9, // padded
            0.1, 0.9, // padded
            0.1, 0.9, // padded
        ];
        let decoded = decode_ctc_greedy_for_input(&values, &[1, 5, 2], &dictionary, 40, 100)
            .expect("decode CTC");
        assert_eq!(decoded.text, "A");
    }

    #[test]
    fn split_preserves_recognizer_aspect_ratio() {
        let image = RgbImage::from_pixel(1_000, 20, Rgb([0, 0, 0]));
        let chunks = split_recognition_crop_for_input(&image, 48, 320).expect("split crop");
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.width() <= 133));
    }

    #[test]
    fn split_uses_a_third_width_overlap_for_chunk_boundaries() {
        let image = RgbImage::from_fn(300, 20, |x, _| Rgb([(x & 0xff) as u8, (x >> 8) as u8, 0]));
        let chunks = split_recognition_crop_for_input(&image, 48, 320).expect("split crop");

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].width(), 133);
        assert_eq!(chunks[1].width(), 133);
        assert_eq!(chunks[2].width(), 122);
        for x in 0..44 {
            assert_eq!(chunks[0].get_pixel(89 + x, 0), chunks[1].get_pixel(x, 0));
        }
    }
}
