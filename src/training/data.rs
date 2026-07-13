use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use image::{GenericImageView, GrayImage};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::training::model::time_steps_for_width;
use crate::training::vocab::Vocab;

#[derive(Debug, Clone)]
struct Sample {
    path: PathBuf,
    text: String,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    samples: Vec<Sample>,
}

#[derive(Debug, Clone)]
pub struct AugmentConfig {
    pub enable: bool,
    pub noise: f32,
    pub blur_prob: f32,
    pub max_brightness: f32,
    pub max_contrast: f32,
    pub erase_prob: f32,
    pub erase_max_fraction: f32,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            enable: false,
            noise: 0.04,
            blur_prob: 0.2,
            max_brightness: 0.1,
            max_contrast: 0.2,
            erase_prob: 0.15,
            erase_max_fraction: 0.3,
        }
    }
}

pub struct Batch {
    pub images: Tensor,
    pub targets: Vec<Vec<usize>>,
    pub texts: Vec<String>,
    pub input_lengths: Vec<usize>,
}

pub struct BatchStats {
    pub total: usize,
    pub skipped: usize,
}

pub struct BatchIter<'a> {
    dataset: &'a Dataset,
    indices: Vec<usize>,
    cursor: usize,
    batch_size: usize,
    device: Device,
    image_height: u32,
    max_width: Option<u32>,
    pad_value: f32,
    vocab: &'a Vocab,
    augment: AugmentConfig,
    rng: StdRng,
}

impl Dataset {
    pub fn from_tsv(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open list {}", path.display()))?;
        let reader = BufReader::new(file);
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        let mut samples = Vec::new();

        for (line_no, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("read line {line_no}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(2, '\t');
            let rel_path = parts
                .next()
                .with_context(|| format!("missing path at line {line_no}"))?;
            let text = parts
                .next()
                .with_context(|| format!("missing label at line {line_no}"))?;

            let path = Path::new(rel_path);
            let full_path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                base.join(path)
            };
            samples.push(Sample {
                path: full_path,
                text: text.to_string(),
            });
        }

        Ok(Self { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn batch_iter<'a>(
        &'a self,
        batch_size: usize,
        shuffle: bool,
        device: Device,
        image_height: u32,
        max_width: Option<u32>,
        pad_value: f32,
        vocab: &'a Vocab,
        bucketed: bool,
        bucket_size: usize,
        seed: u64,
        augment: AugmentConfig,
    ) -> Result<(BatchIter<'a>, BatchStats)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let (mut indexed_widths, stats) = build_indices(self, image_height, max_width, vocab)?;
        if bucketed {
            indexed_widths.sort_by_key(|(_, width)| *width);
            if shuffle {
                let bucket = bucket_size.max(batch_size).max(1);
                for chunk in indexed_widths.chunks_mut(bucket) {
                    chunk.shuffle(&mut rng);
                }
            }
        } else if shuffle {
            indexed_widths.shuffle(&mut rng);
        }

        let iter = BatchIter {
            dataset: self,
            indices: indexed_widths.into_iter().map(|(idx, _)| idx).collect(),
            cursor: 0,
            batch_size,
            device,
            image_height,
            max_width,
            pad_value,
            vocab,
            augment,
            rng,
        };
        Ok((iter, stats))
    }
}

impl<'a> Iterator for BatchIter<'a> {
    type Item = Result<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.indices.len() {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.cursor..end];
        self.cursor = end;
        Some(build_batch(
            self.dataset,
            batch_indices,
            &self.device,
            self.image_height,
            self.max_width,
            self.pad_value,
            self.vocab,
            &self.augment,
            &mut self.rng,
        ))
    }
}

fn build_indices(
    dataset: &Dataset,
    image_height: u32,
    max_width: Option<u32>,
    vocab: &Vocab,
) -> Result<(Vec<(usize, usize)>, BatchStats)> {
    let mut indices = Vec::with_capacity(dataset.samples.len());
    let mut skipped = 0usize;

    for (idx, sample) in dataset.samples.iter().enumerate() {
        let width = match scaled_width_for_sample(sample, image_height, max_width) {
            Ok(value) => value,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let time_steps = time_steps_for_width(width);
        let label_len = sample.text.chars().count();
        if label_len == 0 {
            skipped += 1;
            continue;
        }
        let encoded = match vocab.encode(&sample.text) {
            Ok(encoded) => encoded,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let min_steps = min_steps_for_ctc(&encoded);
        if time_steps < min_steps {
            skipped += 1;
            continue;
        }
        indices.push((idx, width));
    }

    Ok((
        indices,
        BatchStats {
            total: dataset.samples.len(),
            skipped,
        },
    ))
}

fn build_batch(
    dataset: &Dataset,
    indices: &[usize],
    device: &Device,
    image_height: u32,
    max_width: Option<u32>,
    pad_value: f32,
    vocab: &Vocab,
    augment: &AugmentConfig,
    rng: &mut StdRng,
) -> Result<Batch> {
    let mut image_data = Vec::with_capacity(indices.len());
    let mut widths = Vec::with_capacity(indices.len());
    let mut targets = Vec::with_capacity(indices.len());
    let mut texts = Vec::with_capacity(indices.len());
    let mut input_lengths = Vec::with_capacity(indices.len());

    for &idx in indices {
        let sample = &dataset.samples[idx];
        let (data, width) = load_image(sample, image_height, max_width, augment, rng)?;
        let encoded = vocab.encode(&sample.text)?;
        image_data.push(data);
        widths.push(width);
        let input_len = time_steps_for_width(width);
        targets.push(encoded);
        texts.push(sample.text.clone());
        input_lengths.push(input_len);
    }

    let max_width = widths.iter().copied().max().unwrap_or(1);
    let max_width = align_width(max_width, 16);
    let height = image_height as usize;
    let batch = indices.len();

    let mut batch_data = vec![pad_value; batch * height * max_width];
    for (i, data) in image_data.iter().enumerate() {
        let width = widths[i];
        for row in 0..height {
            let src_start = row * width;
            let dst_start = (i * height + row) * max_width;
            batch_data[dst_start..dst_start + width]
                .copy_from_slice(&data[src_start..src_start + width]);
        }
    }

    let images = Tensor::from_vec(batch_data, (batch, 1, height, max_width), device)?;

    Ok(Batch {
        images,
        targets,
        texts,
        input_lengths,
    })
}

fn load_image(
    sample: &Sample,
    height: u32,
    max_width: Option<u32>,
    augment: &AugmentConfig,
    rng: &mut StdRng,
) -> Result<(Vec<f32>, usize)> {
    let img = image::open(&sample.path)
        .with_context(|| format!("open image {}", sample.path.display()))?;
    let mut gray = img.to_luma8();

    if augment.enable {
        gray = apply_augment(gray, augment, rng);
    }

    let (w, h) = gray.dimensions();
    let new_w = scaled_width(w, h, height, max_width);
    let resized = image::imageops::resize(&gray, new_w, height, FilterType::Triangle);

    let mut data = Vec::with_capacity((height * new_w) as usize);
    for pixel in resized.pixels() {
        let v = pixel[0] as f32 / 255.0;
        let v = (v - 0.5) / 0.5;
        data.push(v);
    }

    Ok((data, new_w as usize))
}

fn scaled_width_for_sample(sample: &Sample, height: u32, max_width: Option<u32>) -> Result<usize> {
    let (w, h) = match image::image_dimensions(&sample.path) {
        Ok(dim) => dim,
        Err(_) => {
            let img = image::open(&sample.path)
                .with_context(|| format!("open image {} for dimensions", sample.path.display()))?;
            img.dimensions()
        }
    };
    Ok(scaled_width(w, h, height, max_width) as usize)
}

fn scaled_width(orig_w: u32, orig_h: u32, height: u32, max_width: Option<u32>) -> u32 {
    let scale = height as f32 / orig_h.max(1) as f32;
    let mut new_w = ((orig_w as f32) * scale).round().max(1.0) as u32;
    if let Some(max_width) = max_width {
        if max_width > 0 {
            new_w = new_w.min(max_width);
        }
    }
    new_w
}

fn align_width(width: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return width;
    }
    (width + multiple - 1) / multiple * multiple
}

fn min_steps_for_ctc(encoded: &[usize]) -> usize {
    if encoded.is_empty() {
        return 0;
    }
    let mut repeats = 0usize;
    for i in 1..encoded.len() {
        if encoded[i] == encoded[i - 1] {
            repeats += 1;
        }
    }
    encoded.len() + repeats
}

fn apply_augment(mut img: GrayImage, cfg: &AugmentConfig, rng: &mut StdRng) -> GrayImage {
    let (w, h) = img.dimensions();

    if rng.r#gen::<f32>() < cfg.blur_prob {
        let sigma = rng.gen_range(0.3..1.2);
        img = image::imageops::blur(&img, sigma);
    }

    if rng.r#gen::<f32>() < cfg.erase_prob {
        let frac = rng.gen_range(0.05..cfg.erase_max_fraction.max(0.05));
        let erase_w = (w as f32 * frac).max(1.0) as u32;
        let erase_h = (h as f32 * frac).max(1.0) as u32;
        let x0 = rng.gen_range(0..(w - erase_w + 1).max(1));
        let y0 = rng.gen_range(0..(h - erase_h + 1).max(1));
        for y in y0..(y0 + erase_h).min(h) {
            for x in x0..(x0 + erase_w).min(w) {
                img.put_pixel(x, y, image::Luma([128u8]));
            }
        }
    }

    let brightness = rng.gen_range(-cfg.max_brightness..=cfg.max_brightness);
    let contrast = rng.gen_range(1.0 - cfg.max_contrast..=1.0 + cfg.max_contrast);
    let noise = cfg.noise;

    for pixel in img.pixels_mut() {
        let mut v = pixel[0] as f32 / 255.0;
        v = (v - 0.5) * contrast + 0.5 + brightness;
        if noise > 0.0 {
            v += rng.gen_range(-noise..=noise);
        }
        v = v.clamp(0.0, 1.0);
        pixel[0] = (v * 255.0).round() as u8;
    }

    img
}
