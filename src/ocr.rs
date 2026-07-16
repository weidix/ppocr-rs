use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
#[cfg(feature = "gpu")]
use std::path::PathBuf;
use std::{collections::VecDeque, fs, path::Path};

#[cfg(feature = "cpu")]
use crate::cpu::{CpuOptions, Detector, Recognizer, Tensor};
#[cfg(feature = "cpu")]
use crate::preprocess::prepare_detector;
#[cfg(any(feature = "cpu", feature = "gpu"))]
use crate::preprocess::prepare_recognizer;
use crate::{ModelSize, ModelStore, RgbImage};
#[cfg(feature = "gpu")]
use std::cell::RefCell;

pub const RECOGNIZER_INPUT_HEIGHT: usize = 48;
pub const RECOGNIZER_INPUT_WIDTH: usize = 320;

const MAX_RECTIFIED_EDGE: f32 = 4_096.0;
const MAX_RECTIFIED_PIXELS: f32 = 8_000_000.0;
const RECOGNIZER_BATCH_SIZE: usize = 6;
const PROBABILITY_EPSILON: f32 = 1e-8;
const DETECTOR_LIMIT_SIDE: f64 = 736.0;
const DETECTOR_MAX_SIDE: f64 = 4_000.0;
#[cfg(feature = "gpu")]
const GPU_DETECTOR_CACHE_LIMIT: usize = 2;
#[cfg(feature = "gpu")]
const GPU_RECOGNIZER_CACHE_LIMIT: usize = 4;

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

/// Backend-independent detector input geometry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DetectorInputPlan {
    input_width: usize,
    input_height: usize,
    transform: DetectorTransform,
}

impl DetectorInputPlan {
    pub(crate) fn new(image: &RgbImage, max_side: Option<u32>) -> Result<Self> {
        let width = image.width();
        let height = image.height();
        let ratio = match max_side {
            Some(limit) if limit > 0 => (f64::from(limit) / f64::from(width.max(height))).min(1.0),
            Some(_) => bail!("detector maximum side must be positive"),
            None => default_detector_ratio(width, height),
        };
        let input_width = aligned_dimension(f64::from(width) * ratio)?;
        let input_height = aligned_dimension(f64::from(height) * ratio)?;
        Ok(Self {
            input_width: input_width as usize,
            input_height: input_height as usize,
            transform: DetectorTransform::new(width, height, input_width, input_height)?,
        })
    }

    pub(crate) const fn input_width(self) -> usize {
        self.input_width
    }

    pub(crate) const fn input_height(self) -> usize {
        self.input_height
    }

    pub(crate) const fn transform(self) -> DetectorTransform {
        self.transform
    }

    pub(crate) fn corners(self) -> [Point; 4] {
        [
            Point(0.0, 0.0),
            Point(self.transform.source_width as f32, 0.0),
            Point(
                self.transform.source_width as f32,
                self.transform.source_height as f32,
            ),
            Point(0.0, self.transform.source_height as f32),
        ]
    }
}

/// Backend-independent recognizer input geometry for one text segment.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecognitionInputPlan {
    corners: [Point; 4],
    input_width: usize,
    content_width: usize,
}

impl RecognitionInputPlan {
    pub(crate) const fn corners(self) -> [Point; 4] {
        self.corners
    }

    pub(crate) const fn input_width(self) -> usize {
        self.input_width
    }

    pub(crate) const fn input_height(self) -> usize {
        RECOGNIZER_INPUT_HEIGHT
    }

    pub(crate) const fn content_width(self) -> usize {
        self.content_width
    }
}

pub(crate) fn recognition_input_plan(
    polygon: [Point; 4],
    input_width: usize,
) -> Result<RecognitionInputPlan> {
    let width = distance(polygon[0], polygon[1]).max(distance(polygon[3], polygon[2]));
    let height = distance(polygon[0], polygon[3]).max(distance(polygon[1], polygon[2]));
    let (width, height) = bounded_crop_dimensions(width, height)?;
    let content_width = scaled_recognizer_width(width, height)?;
    ensure!(
        content_width <= input_width,
        "recognizer crop exceeds its batch canvas"
    );
    Ok(RecognitionInputPlan {
        corners: polygon,
        input_width,
        content_width,
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

fn scaled_recognizer_width(width: u32, height: u32) -> Result<usize> {
    let scaled = (RECOGNIZER_INPUT_HEIGHT as u32)
        .checked_mul(width)
        .context("recognizer content width overflow")?
        .div_ceil(height)
        .max(1);
    Ok(scaled as usize)
}

fn recognizer_batch_width(polygons: &[[Point; 4]]) -> Result<usize> {
    let maximum = polygons
        .iter()
        .try_fold(RECOGNIZER_INPUT_WIDTH, |maximum, polygon| {
            let width = distance(polygon[0], polygon[1]).max(distance(polygon[3], polygon[2]));
            let height = distance(polygon[0], polygon[3]).max(distance(polygon[1], polygon[2]));
            let (width, height) = bounded_crop_dimensions(width, height)?;
            Ok::<_, anyhow::Error>(maximum.max(scaled_recognizer_width(width, height)?))
        })?;
    maximum
        .checked_add(3)
        .map(|width| width / 4 * 4)
        .context("recognizer batch width overflow")
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

/// Inference device used by the end-to-end OCR engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OcrBackend {
    /// Native CPU kernels.
    Cpu,
    /// WGPU compute kernels.
    Gpu,
}

/// Configuration for the end-to-end OCR pipeline.
#[derive(Clone, Debug)]
pub struct OcrOptions {
    /// Inference device.
    pub backend: OcrBackend,
    /// Detector model tier.
    pub detector_size: ModelSize,
    /// Recognizer model tier.
    pub recognizer_size: ModelSize,
    /// CPU worker threads. The GPU backend ignores this value.
    pub threads: usize,
    /// Detector output postprocessing settings.
    pub detector_postprocess: DetectorPostprocessOptions,
    /// Optional maximum detector image side. `None` uses the released model's
    /// default resize policy.
    pub detector_max_side: Option<u32>,
}

impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            backend: OcrBackend::Cpu,
            detector_size: ModelSize::Tiny,
            recognizer_size: ModelSize::Tiny,
            threads: std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(4),
            detector_postprocess: DetectorPostprocessOptions::default(),
            detector_max_side: None,
        }
    }
}

impl OcrOptions {
    /// Validates options before model resolution or inference.
    pub fn validate(&self) -> Result<()> {
        self.detector_postprocess.validate()?;
        ensure!(self.threads > 0, "worker thread count must be positive");
        if let Some(max_side) = self.detector_max_side {
            ensure!(max_side > 0, "detector maximum side must be positive");
        }
        Ok(())
    }
}

/// One recognized text region.
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
#[derive(Clone, Debug, Serialize)]
pub struct OcrResult {
    /// Source image dimensions as `[width, height]`.
    pub source_size: [u32; 2],
    /// Detector input dimensions as `[width, height]` after resizing.
    pub detector_input_size: [usize; 2],
    /// Text regions in natural reading order.
    pub lines: Vec<OcrLine>,
}

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

/// An end-to-end PP-OCRv6 inference engine.
pub struct OcrEngine {
    runtime: OcrRuntime,
    dictionary: Vec<String>,
    options: OcrOptions,
}

enum OcrRuntime {
    #[cfg(feature = "cpu")]
    Cpu {
        detector: Detector,
        recognizer: Recognizer,
    },
    #[cfg(feature = "gpu")]
    Gpu(GpuOcrRuntime),
    #[cfg(not(any(feature = "cpu", feature = "gpu")))]
    Unavailable,
}

#[cfg(feature = "gpu")]
struct GpuOcrRuntime {
    gpu: crate::gpu::Gpu,
    detector_model: PathBuf,
    detector_size: ModelSize,
    detector_cache: RefCell<Vec<([usize; 4], crate::gpu::Detector)>>,
    recognizer_model: PathBuf,
    recognizer_size: ModelSize,
    recognizer_cache: RefCell<Vec<([usize; 4], crate::gpu::Recognizer)>>,
}

impl OcrEngine {
    /// Downloads missing pinned models through `store`, then loads an OCR engine.
    pub fn load_from_store(store: &ModelStore, options: OcrOptions) -> Result<Self> {
        #[cfg(not(any(feature = "cpu", feature = "gpu")))]
        {
            let _ = (store, options);
            bail!("no OCR backend is compiled; rebuild with --features cpu or --features gpu")
        }
        #[cfg(any(feature = "cpu", feature = "gpu"))]
        {
            let paths = store.ensure_pair(options.detector_size, options.recognizer_size)?;
            Self::load(
                &paths.detector.weights,
                &paths.recognizer.weights,
                &paths.recognizer.inference,
                options,
            )
        }
    }

    /// Loads an engine from explicit detector, recognizer, and dictionary paths.
    pub fn load(
        detector_model: impl AsRef<Path>,
        recognizer_model: impl AsRef<Path>,
        dictionary: impl AsRef<Path>,
        options: OcrOptions,
    ) -> Result<Self> {
        #[cfg(not(any(feature = "cpu", feature = "gpu")))]
        {
            let _ = (detector_model, recognizer_model, dictionary, options);
            bail!("no OCR backend is compiled; rebuild with --features cpu or --features gpu")
        }
        #[cfg(any(feature = "cpu", feature = "gpu"))]
        {
            options.validate()?;
            let detector_model = detector_model.as_ref();
            let recognizer_model = recognizer_model.as_ref();
            let dictionary_path = dictionary.as_ref();
            let dictionary = load_dictionary(dictionary_path)?;
            validate_recognizer_dictionary(&dictionary, options.recognizer_size)?;
            let runtime = match options.backend {
                OcrBackend::Cpu => {
                    #[cfg(feature = "cpu")]
                    {
                        let cpu = CpuOptions {
                            threads: options.threads,
                        };
                        let detector = Detector::load(detector_model, options.detector_size, cpu)
                            .with_context(|| {
                            format!("load detector {}", detector_model.display())
                        })?;
                        let recognizer =
                            Recognizer::load(recognizer_model, options.recognizer_size, cpu)
                                .with_context(|| {
                                    format!("load recognizer {}", recognizer_model.display())
                                })?;
                        OcrRuntime::Cpu {
                            detector,
                            recognizer,
                        }
                    }
                    #[cfg(not(feature = "cpu"))]
                    {
                        bail!("the CPU backend is not compiled; rebuild with --features cpu")
                    }
                }
                OcrBackend::Gpu => {
                    #[cfg(feature = "gpu")]
                    {
                        let gpu = crate::gpu::Gpu::new().context("initialize GPU backend")?;
                        OcrRuntime::Gpu(GpuOcrRuntime {
                            gpu,
                            detector_model: detector_model.to_path_buf(),
                            detector_size: options.detector_size,
                            detector_cache: RefCell::new(Vec::new()),
                            recognizer_model: recognizer_model.to_path_buf(),
                            recognizer_size: options.recognizer_size,
                            recognizer_cache: RefCell::new(Vec::new()),
                        })
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        bail!("the GPU backend is not compiled; rebuild with --features gpu")
                    }
                }
            };
            Ok(Self {
                runtime,
                dictionary,
                options,
            })
        }
    }

    /// Decodes an image file into text regions.
    pub fn recognize_path(&self, path: impl AsRef<Path>) -> Result<OcrResult> {
        let image = RgbImage::from_path(path)?;
        self.recognize(&image)
    }

    /// Decodes an RGB image into text regions.
    pub fn recognize(&self, image: &RgbImage) -> Result<OcrResult> {
        match &self.runtime {
            #[cfg(feature = "cpu")]
            OcrRuntime::Cpu {
                detector,
                recognizer,
            } => recognize_with(
                image,
                &self.dictionary,
                &self.options,
                |plan| {
                    let prepared = detector.with_thread_pool(|| prepare_detector(image, plan));
                    let output =
                        detector.forward(Tensor::from_f32(prepared.shape(), prepared.data)?)?;
                    Ok((output.as_f32()?.to_vec(), output.shape().to_vec()))
                },
                |plans| {
                    let prepared = recognizer.with_thread_pool(|| prepare_recognizer(image, plans));
                    let output =
                        recognizer.forward(Tensor::from_f32(prepared.shape(), prepared.data)?)?;
                    Ok((output.as_f32()?.to_vec(), output.shape().to_vec()))
                },
            ),
            #[cfg(feature = "gpu")]
            OcrRuntime::Gpu(runtime) => {
                let gpu_image = runtime
                    .gpu
                    .upload_rgb(image.width(), image.height(), image.pixels())
                    .context("upload source image to GPU")?;
                recognize_with(
                    image,
                    &self.dictionary,
                    &self.options,
                    |plan| runtime.run_detector(&gpu_image, plan),
                    |plans| runtime.run_recognizer(image, plans),
                )
            }
            #[cfg(not(any(feature = "cpu", feature = "gpu")))]
            OcrRuntime::Unavailable => {
                bail!("no OCR backend is compiled; rebuild with --features cpu or --features gpu")
            }
        }
    }
}

fn recognize_with(
    image: &RgbImage,
    dictionary: &[String],
    options: &OcrOptions,
    mut run_detector: impl FnMut(DetectorInputPlan) -> Result<(Vec<f32>, Vec<usize>)>,
    mut run_recognizer: impl FnMut(&[RecognitionInputPlan]) -> Result<(Vec<f32>, Vec<usize>)>,
) -> Result<OcrResult> {
    let detector_plan = DetectorInputPlan::new(image, options.detector_max_side)?;
    let detector_input_size = [detector_plan.input_width(), detector_plan.input_height()];
    let (detector_values, detector_shape) = run_detector(detector_plan)?;
    let detections = extract_detections(
        &detector_values,
        &detector_shape,
        detector_plan.transform(),
        options.detector_postprocess,
    )?;

    let mut recognition_order = (0..detections.len()).collect::<Vec<_>>();
    recognition_order.sort_by(|&left, &right| {
        polygon_aspect_ratio(detections[left].polygon)
            .total_cmp(&polygon_aspect_ratio(detections[right].polygon))
    });
    let mut decoded = vec![
        DecodedText {
            text: String::new(),
            score: 0.0
        };
        detections.len()
    ];
    for indices in recognition_order.chunks(RECOGNIZER_BATCH_SIZE) {
        let polygons = indices
            .iter()
            .map(|&index| detections[index].polygon)
            .collect::<Vec<_>>();
        let input_width = recognizer_batch_width(&polygons)?;
        let plans = polygons
            .into_iter()
            .map(|polygon| recognition_input_plan(polygon, input_width))
            .collect::<Result<Vec<_>>>()?;
        let (values, shape) = run_recognizer(&plans)?;
        let batch = decode_ctc_batch(&values, &shape, dictionary, &plans)?;
        for (&index, result) in indices.iter().zip(batch) {
            decoded[index] = result;
        }
    }

    let lines = detections
        .into_iter()
        .zip(decoded)
        .map(|(detection, decoded)| OcrLine {
            polygon: detection.polygon,
            detection_score: detection.score,
            text: decoded.text,
            recognition_score: decoded.score,
        })
        .collect();

    Ok(OcrResult {
        source_size: [image.width(), image.height()],
        detector_input_size,
        lines,
    })
}

#[cfg(feature = "gpu")]
impl GpuOcrRuntime {
    fn run_detector(
        &self,
        image: &crate::gpu::GpuImage,
        plan: DetectorInputPlan,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let shape = [1, 3, plan.input_height(), plan.input_width()];
        let mut cache = self
            .detector_cache
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("GPU detector cache is already in use"))?;
        let index = match cache.iter().position(|(cached, _)| *cached == shape) {
            Some(index) => index,
            None => {
                if cache.len() == GPU_DETECTOR_CACHE_LIMIT {
                    cache.remove(0);
                }
                let detector = crate::gpu::Detector::load(
                    &self.gpu,
                    &self.detector_model,
                    self.detector_size,
                    shape,
                )
                .with_context(|| {
                    format!(
                        "load GPU detector {} for input {:?}",
                        self.detector_model.display(),
                        shape
                    )
                })?;
                cache.push((shape, detector));
                cache.len() - 1
            }
        };
        let output = cache[index].1.forward_image(
            image,
            crate::gpu::ImagePreprocess::detector(plan.corners().map(point_coordinates)),
        )?;
        Ok((output.values, output.shape))
    }

    fn run_recognizer(
        &self,
        image: &RgbImage,
        plans: &[RecognitionInputPlan],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let prepared = prepare_recognizer(image, plans);
        let shape = prepared.shape();
        let mut cache = self
            .recognizer_cache
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("GPU recognizer cache is already in use"))?;
        let index = match cache.iter().position(|(cached, _)| *cached == shape) {
            Some(index) => index,
            None => {
                if cache.len() == GPU_RECOGNIZER_CACHE_LIMIT {
                    cache.remove(0);
                }
                let recognizer = crate::gpu::Recognizer::load(
                    &self.gpu,
                    &self.recognizer_model,
                    self.recognizer_size,
                    shape,
                )
                .with_context(|| {
                    format!(
                        "load GPU recognizer {} for input {:?}",
                        self.recognizer_model.display(),
                        shape
                    )
                })?;
                cache.push((shape, recognizer));
                cache.len() - 1
            }
        };
        let output = cache[index].1.forward(&prepared.data)?;
        Ok((output.values, output.shape))
    }
}

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

fn decode_ctc_batch(
    values: &[f32],
    shape: &[usize],
    dictionary: &[String],
    plans: &[RecognitionInputPlan],
) -> Result<Vec<DecodedText>> {
    if shape.len() != 3 || shape[0] != plans.len() || shape[1] == 0 || shape[2] < 2 {
        bail!(
            "recognizer output shape {shape:?}, expected [{}, time, classes>=2]",
            plans.len()
        );
    }
    let item_len = shape[1]
        .checked_mul(shape[2])
        .context("recognizer output item size overflow")?;
    let expected = plans
        .len()
        .checked_mul(item_len)
        .context("recognizer output size overflow")?;
    if values.len() != expected {
        bail!(
            "recognizer output has {} values, expected {expected}",
            values.len()
        );
    }
    plans
        .iter()
        .zip(values.chunks_exact(item_len))
        .map(|(plan, item)| {
            decode_ctc_greedy_for_input(
                item,
                &[1, shape[1], shape[2]],
                dictionary,
                plan.content_width(),
                plan.input_width(),
            )
        })
        .collect()
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

fn polygon_aspect_ratio(polygon: [Point; 4]) -> f32 {
    let width = distance(polygon[0], polygon[1]).max(distance(polygon[3], polygon[2]));
    let height = distance(polygon[0], polygon[3]).max(distance(polygon[1], polygon[2]));
    width / height.max(f32::EPSILON)
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

#[cfg(feature = "gpu")]
fn point_coordinates(point: Point) -> [f32; 2] {
    [point.0, point.1]
}

fn distance(left: Point, right: Point) -> f32 {
    ((left.0 - right.0).powi(2) + (left.1 - right.1).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn recognition_plan_preserves_the_entire_crop() {
        let polygon = [
            Point(0.0, 0.0),
            Point(1_000.0, 0.0),
            Point(1_000.0, 20.0),
            Point(0.0, 20.0),
        ];
        let width = recognizer_batch_width(&[polygon]).expect("batch width");
        let plan = recognition_input_plan(polygon, width).expect("recognition plan");
        assert_eq!(plan.corners(), polygon);
        assert_eq!(plan.content_width(), 2_400);
        assert_eq!(plan.input_width(), 2_400);
    }

    #[test]
    fn recognizer_batch_uses_the_largest_aspect_ratio() {
        let narrow = [
            Point(0.0, 0.0),
            Point(80.0, 0.0),
            Point(80.0, 20.0),
            Point(0.0, 20.0),
        ];
        let wide = [
            Point(0.0, 0.0),
            Point(300.0, 0.0),
            Point(300.0, 20.0),
            Point(0.0, 20.0),
        ];
        assert_eq!(
            recognizer_batch_width(&[narrow, wide]).expect("batch width"),
            720
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_recognizer_accepts_a_real_batch() {
        let model = Path::new("models/tiny-rec/model.safetensors");
        if !model.is_file() {
            return;
        }
        let recognizer = Recognizer::load(model, ModelSize::Tiny, CpuOptions { threads: 1 })
            .expect("load CPU recognizer");
        let input = Tensor::from_f32(
            [2, 3, RECOGNIZER_INPUT_HEIGHT, RECOGNIZER_INPUT_WIDTH],
            vec![0.0; 2 * 3 * RECOGNIZER_INPUT_HEIGHT * RECOGNIZER_INPUT_WIDTH],
        )
        .expect("batch input");
        let output = recognizer.forward(input).expect("recognize batch");
        assert_eq!(output.shape()[0], 2);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_recognizer_accepts_a_real_batch() {
        let model = Path::new("models/tiny-rec/model.safetensors");
        if !model.is_file() {
            return;
        }
        let Ok(gpu) = crate::gpu::Gpu::new() else {
            return;
        };
        let shape = [2, 3, RECOGNIZER_INPUT_HEIGHT, RECOGNIZER_INPUT_WIDTH];
        let recognizer = crate::gpu::Recognizer::load(&gpu, model, ModelSize::Tiny, shape)
            .expect("load GPU recognizer");
        let output = recognizer
            .forward(&vec![0.0; shape.into_iter().product()])
            .expect("recognize batch");
        assert_eq!(output.shape[0], 2);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_engine_runs_device_preprocessing_when_local_models_are_available() {
        let detector = Path::new("models/tiny-det/model.safetensors");
        let recognizer = Path::new("models/tiny-rec/model.safetensors");
        let dictionary = Path::new("models/tiny-rec/inference.yml");
        if !detector.is_file() || !recognizer.is_file() || !dictionary.is_file() {
            return;
        }
        if crate::gpu::Gpu::new().is_err() {
            return;
        }

        let engine = OcrEngine::load(
            detector,
            recognizer,
            dictionary,
            OcrOptions {
                backend: OcrBackend::Gpu,
                detector_max_side: Some(32),
                ..OcrOptions::default()
            },
        )
        .expect("load GPU OCR engine");
        let image = RgbImage::new(32, 32, vec![255; 32 * 32 * 3]).expect("source image");
        let result = engine.recognize(&image).expect("run GPU OCR");
        assert_eq!(result.source_size, [32, 32]);
        assert_eq!(result.detector_input_size, [32, 32]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_engine_runs_shared_preprocessing_plan_when_local_models_are_available() {
        let detector = Path::new("models/tiny-det/model.safetensors");
        let recognizer = Path::new("models/tiny-rec/model.safetensors");
        let dictionary = Path::new("models/tiny-rec/inference.yml");
        if !detector.is_file() || !recognizer.is_file() || !dictionary.is_file() {
            return;
        }

        let engine = OcrEngine::load(
            detector,
            recognizer,
            dictionary,
            OcrOptions {
                backend: OcrBackend::Cpu,
                detector_max_side: Some(32),
                threads: 1,
                ..OcrOptions::default()
            },
        )
        .expect("load CPU OCR engine");
        let image = RgbImage::new(32, 32, vec![255; 32 * 32 * 3]).expect("source image");
        let result = engine.recognize(&image).expect("run CPU OCR");
        assert_eq!(result.source_size, [32, 32]);
        assert_eq!(result.detector_input_size, [32, 32]);
    }
}
