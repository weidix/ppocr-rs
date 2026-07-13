use crate::burn_runtime::preprocess::{DetectorTransform, RECOGNIZER_SHAPE};
use anyhow::{Context, Result, bail};
use image::{Rgb, RgbImage, imageops};
use serde::Serialize;
use std::{collections::VecDeque, fs, path::Path};

const MAX_RECTIFIED_EDGE: f32 = 4_096.0;
const MAX_RECTIFIED_PIXELS: f32 = 8_000_000.0;
const MAX_RECOGNITION_CHUNKS: usize = 64;
const PROBABILITY_EPSILON: f32 = 1e-8;

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

pub fn split_recognition_crop(image: &RgbImage) -> Result<Vec<RgbImage>> {
    if image.width() == 0 || image.height() == 0 {
        bail!("cannot split an empty recognition crop");
    }
    let max_width = (image.height() as usize)
        .checked_mul(RECOGNIZER_SHAPE[3])
        .context("recognition crop width overflow")?
        / RECOGNIZER_SHAPE[2];
    let max_width = u32::try_from(max_width.max(1)).context("recognition crop width overflow")?;
    if image.width() <= max_width {
        return Ok(vec![image.clone()]);
    }

    let overlap = if max_width > 1 {
        (max_width / 6).clamp(1, max_width - 1)
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
    if content_width == 0 || content_height == 0 {
        bail!("detector transform has an empty content region");
    }

    let mut visited = vec![false; content_width * content_height];
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
            let polygon = fit_rotated_box(&component.points, options.unclip_ratio);
            let polygon = polygon.map(|point| {
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
    let entries = parse_dictionary(&contents)
        .with_context(|| format!("parse recognition dictionary {}", path.display()))?;
    Ok(entries)
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
            let trimmed_start = entry_line.trim_start();
            let Some(value) = trimmed_start.strip_prefix("- ") else {
                let entry_indentation = entry_line.len() - trimmed_start.len();
                if entry_indentation <= indentation {
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

pub fn decode_ctc_greedy(
    values: &[f32],
    shape: &[usize],
    dictionary: &[String],
    content_width: usize,
) -> Result<DecodedText> {
    let (time_steps, classes) = recognizer_output_shape(shape, values.len())?;
    let implicit_space_class = classes == dictionary.len().saturating_add(2);
    if classes != dictionary.len() + 1 && !implicit_space_class {
        bail!(
            "recognizer output has {classes} classes, but the dictionary has {} entries; expected dictionary entries plus the CTC blank class",
            dictionary.len()
        );
    }
    if content_width == 0 || content_width > RECOGNIZER_SHAPE[3] {
        bail!(
            "recognizer content width {content_width} is outside 1..={}",
            RECOGNIZER_SHAPE[3]
        );
    }
    if values.iter().any(|value| !value.is_finite()) {
        bail!("recognizer output contains non-finite values");
    }

    let valid_steps = (content_width
        .checked_mul(time_steps)
        .context("recognizer time-step calculation overflow")?
        + RECOGNIZER_SHAPE[3]
        - 1)
        / RECOGNIZER_SHAPE[3];
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

fn validate_probability(value: f32, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("{name} must be a finite value between zero and one");
    }
    Ok(())
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
        let value = values[y * output_width + x];
        points.push(Point(x as f32 + 0.5, y as f32 + 0.5));
        score_sum += f64::from(value);
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
            if visited[next_index] || values[next_y * output_width + next_x] < threshold {
                continue;
            }
            visited[next_index] = true;
            queue.push_back((next_x, next_y));
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
    let width = (width * scale).round().clamp(1.0, MAX_RECTIFIED_EDGE) as u32;
    let height = (height * scale).round().clamp(1.0, MAX_RECTIFIED_EDGE) as u32;
    Ok((width, height))
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

fn argmax(row: &[f32]) -> (usize, f32) {
    row.iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("recognizer output rows are non-empty")
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
    let overlap = (2..=maximum)
        .rev()
        .find(|size| target_chars[target_chars.len() - *size..] == next_chars[..*size])
        .unwrap_or(0);
    target.extend(next_chars[overlap..].iter().copied());
    next_chars.len() - overlap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn_runtime::preprocess::prepare_fixed_detector_with_transform;
    use image::Rgb;

    #[test]
    fn detector_postprocess_ignores_letterbox_padding() {
        let image = RgbImage::from_pixel(400, 400, Rgb([0, 0, 0]));
        let input = prepare_fixed_detector_with_transform(&image).expect("prepare detector");
        let mut values = vec![0.0; DETECTOR_SHAPE_AREA];
        values[20 * 736 + 500] = 0.9;

        let detections = extract_detections(
            &values,
            &[1, 1, 416, 736],
            input.transform,
            DetectorPostprocessOptions {
                min_area: 1,
                unclip_ratio: 1.0,
                ..Default::default()
            },
        )
        .expect("postprocess detector");

        assert!(detections.is_empty());
    }

    #[test]
    fn detector_postprocess_maps_a_component_back_to_source_coordinates() {
        let image = RgbImage::from_pixel(736, 416, Rgb([0, 0, 0]));
        let input = prepare_fixed_detector_with_transform(&image).expect("prepare detector");
        let mut values = vec![0.0; DETECTOR_SHAPE_AREA];
        for y in 20..22 {
            for x in 10..14 {
                values[y * 736 + x] = 0.9;
            }
        }

        let detections = extract_detections(
            &values,
            &[1, 1, 416, 736],
            input.transform,
            DetectorPostprocessOptions {
                min_area: 4,
                unclip_ratio: 1.0,
                ..Default::default()
            },
        )
        .expect("postprocess detector");

        assert_eq!(detections.len(), 1);
        assert!((detections[0].score - 0.9).abs() < 1e-6);
        let xs = detections[0].polygon.map(|point| point.0);
        let ys = detections[0].polygon.map(|point| point.1);
        assert!((xs.iter().copied().fold(f32::INFINITY, f32::min) - 10.0).abs() < 1e-4);
        assert!((xs.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 14.0).abs() < 1e-4);
        assert!((ys.iter().copied().fold(f32::INFINITY, f32::min) - 20.0).abs() < 1e-4);
        assert!((ys.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 22.0).abs() < 1e-4);
    }

    #[test]
    fn rectification_preserves_an_axis_aligned_crop() {
        let mut image = RgbImage::new(4, 2);
        for y in 0..2 {
            for x in 0..4 {
                image.put_pixel(x, y, Rgb([(x + y * 4) as u8, 0, 0]));
            }
        }
        let crop = rectify_text_crop(
            &image,
            [
                Point(0.0, 0.0),
                Point(4.0, 0.0),
                Point(4.0, 2.0),
                Point(0.0, 2.0),
            ],
        )
        .expect("rectify crop");

        assert_eq!(crop.dimensions(), (4, 2));
        assert_eq!(crop, image);
    }

    #[test]
    fn ctc_decoder_collapses_repetitions_and_honors_blank_tokens() {
        let dictionary = vec!["A".to_owned(), "B".to_owned()];
        let values = [
            0.9, 0.1, 0.0, // blank
            0.1, 0.8, 0.1, // A
            0.1, 0.7, 0.2, // repeated A
            0.8, 0.1, 0.1, // blank
            0.1, 0.1, 0.8, // B
        ];

        let decoded = decode_ctc_greedy(&values, &[1, 5, 3], &dictionary, 320).expect("decode CTC");

        assert_eq!(decoded.text, "AB");
        assert!((decoded.score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn ctc_decoder_rejects_a_dictionary_with_the_wrong_class_count() {
        let error = decode_ctc_greedy(&[1.0, 0.0, 0.0, 0.0], &[1, 1, 4], &["A".to_owned()], 320)
            .expect_err("class count must fail");
        assert!(error.to_string().contains("classes"));
    }

    #[test]
    fn ctc_decoder_supports_the_paddlex_implicit_space_class() {
        let decoded = decode_ctc_greedy(&[0.0, 0.1, 0.9], &[1, 1, 3], &["A".to_owned()], 320)
            .expect("decode implicit space");

        assert_eq!(decoded.text, " ");
        assert!((decoded.score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn long_recognition_crops_are_split_without_aspect_ratio_squashing() {
        let image = RgbImage::from_pixel(1_000, 20, Rgb([0, 0, 0]));
        let chunks = split_recognition_crop(&image).expect("split recognition crop");

        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.width() <= 133));
        assert_eq!(chunks.first().expect("first chunk").width(), 133);
        assert_eq!(chunks.last().expect("last chunk").width(), 112);
    }

    #[test]
    fn decoded_chunks_remove_a_shared_boundary_once() {
        let decoded = join_decoded_texts(&[
            DecodedText {
                text: "hello wor".to_owned(),
                score: 0.8,
            },
            DecodedText {
                text: "world".to_owned(),
                score: 0.9,
            },
        ]);

        assert_eq!(decoded.text, "hello world");
        assert!(decoded.score > 0.8 && decoded.score < 0.9);
    }

    #[test]
    fn dictionary_parser_accepts_a_paddlex_model_config() {
        let dictionary = parse_dictionary(
            "PostProcess:\n  character_dict:\n  - A\n  - ''''\n  - ' '\n  use_space_char: true\n",
        )
        .expect("parse model dictionary");

        assert_eq!(dictionary, ["A", "'", " "]);
    }

    #[test]
    fn dictionary_parser_preserves_unquoted_whitespace_characters() {
        let dictionary = parse_dictionary("PostProcess:\n  character_dict:\n  -  \n  - \u{3000}\n")
            .expect("parse whitespace characters");

        assert_eq!(dictionary, [" ", "\u{3000}"]);
    }

    const DETECTOR_SHAPE_AREA: usize = 416 * 736;
}
