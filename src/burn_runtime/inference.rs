mod detector {
    include!(env!("PPOCR_BURN_DETECTOR_MODEL_RS"));
}

mod recognizer {
    include!(env!("PPOCR_BURN_RECOGNIZER_MODEL_RS"));
}

use anyhow::{Context, Result, bail};
use burn::{
    backend::{
        Metal,
        wgpu::{WgpuDevice, graphics, init_setup},
    },
    tensor::{Tensor, TensorData, backend::Backend},
};
use ppocr_rs::burn_runtime::{
    ocr::{
        DecodedText, Detection, DetectorPostprocessOptions, Point, decode_ctc_greedy_with_width,
        extract_detections, join_decoded_texts, load_dictionary, rectify_text_crop,
        split_recognition_crop_with_width,
    },
    preprocess::{
        PreparedInput, load_rgb, prepare_fixed_detector_with_transform,
        prepare_fixed_recognizer_from_image_with_width, recognizer_shape_for_width,
    },
};
use serde::Serialize;
use std::{env, fs, path::PathBuf};

type B = Metal;

struct Arguments {
    image: PathBuf,
    dictionary: PathBuf,
    output: Option<PathBuf>,
    recognizer_width: usize,
    postprocess: DetectorPostprocessOptions,
}

#[derive(Serialize)]
struct TextResult {
    polygon: [Point; 4],
    detection_score: f32,
    text: String,
    recognition_score: f32,
}

#[derive(Serialize)]
struct OcrReport {
    runtime: &'static str,
    image: String,
    source_shape: [u32; 2],
    detector_input_shape: [usize; 4],
    recognizer_input_shape: [usize; 4],
    detector_binary_threshold: f32,
    detector_box_threshold: f32,
    detector_unclip_ratio: f32,
    texts: Vec<TextResult>,
}

pub fn run() -> Result<()> {
    let arguments = parse_arguments()?;
    arguments.postprocess.validate()?;
    let recognizer_input_shape = recognizer_shape_for_width(arguments.recognizer_width)?;
    let image = load_rgb(&arguments.image)?;
    let source_shape = [image.width(), image.height()];
    let detector_input = prepare_fixed_detector_with_transform(&image)?;
    let dictionary = load_dictionary(&arguments.dictionary)?;

    let device = WgpuDevice::default();
    init_setup::<graphics::Metal>(&device, Default::default());
    let detector = detector::Model::<B>::from_embedded(&device);
    let recognizer = recognizer::Model::<B>::from_embedded(&device);

    let detector_tensor = to_tensor(&detector_input.input, &device);
    let detector_output = read_output("detector", detector.forward(detector_tensor), &device)?;
    let detections = extract_detections(
        &detector_output.values,
        &detector_output.shape,
        detector_input.transform,
        arguments.postprocess,
    )?;

    let mut texts = Vec::with_capacity(detections.len());
    for detection in detections {
        let crop = rectify_text_crop(&image, detection.polygon)?;
        let mut decoded_chunks = Vec::new();
        for crop in split_recognition_crop_with_width(&crop, arguments.recognizer_width)? {
            let recognizer_input =
                prepare_fixed_recognizer_from_image_with_width(&crop, arguments.recognizer_width)?;
            let recognizer_tensor = to_tensor(&recognizer_input.input, &device);
            let recognizer_output =
                read_output("recognizer", recognizer.forward(recognizer_tensor), &device)?;
            decoded_chunks.push(decode_ctc_greedy_with_width(
                &recognizer_output.values,
                &recognizer_output.shape,
                &dictionary,
                recognizer_input.content_width,
                arguments.recognizer_width,
            )?);
        }
        let decoded = join_decoded_texts(&decoded_chunks);
        texts.push(text_result(detection, decoded));
    }

    let report = OcrReport {
        runtime: "burn-0.21-metal-fusion",
        image: arguments.image.display().to_string(),
        source_shape,
        detector_input_shape: detector_input.input.shape(),
        recognizer_input_shape,
        detector_binary_threshold: arguments.postprocess.binary_threshold,
        detector_box_threshold: arguments.postprocess.box_threshold,
        detector_unclip_ratio: arguments.postprocess.unclip_ratio,
        texts,
    };
    write_report(&report, arguments.output.as_ref())
}

fn text_result(detection: Detection, decoded: DecodedText) -> TextResult {
    TextResult {
        polygon: detection.polygon,
        detection_score: detection.score,
        text: decoded.text,
        recognition_score: decoded.score,
    }
}

struct TensorOutput {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn to_tensor(input: &PreparedInput, device: &WgpuDevice) -> Tensor<B, 4> {
    Tensor::from_data(TensorData::new(input.data.clone(), input.shape()), device)
}

fn read_output<const D: usize>(
    name: &str,
    output: Tensor<B, D>,
    device: &WgpuDevice,
) -> Result<TensorOutput> {
    B::sync(device).with_context(|| format!("synchronize {name} output"))?;
    let data = output.into_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|error| anyhow::anyhow!("read {name} f32 output: {error}"))?
        .to_vec();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{name} output contains non-finite values");
    }
    Ok(TensorOutput {
        shape: data.shape.as_slice().to_vec(),
        values,
    })
}

fn write_report(report: &OcrReport, output: Option<&PathBuf>) -> Result<()> {
    let serialized = serde_json::to_string_pretty(report).context("serialize OCR output")?;
    match output {
        Some(path) => fs::write(path, serialized)
            .with_context(|| format!("write OCR output {}", path.display())),
        None => {
            println!("{serialized}");
            Ok(())
        }
    }
}

fn parse_arguments() -> Result<Arguments> {
    let mut image = None;
    let mut dictionary = None;
    let mut output = None;
    let embedded_width = embedded_recognizer_width();
    let mut recognizer_width = embedded_width;
    let mut postprocess = DetectorPostprocessOptions::default();
    let mut arguments = env::args_os().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.to_string_lossy().as_ref() {
            "--image" => image = Some(PathBuf::from(next_value(&mut arguments, "--image")?)),
            "--dict" => dictionary = Some(PathBuf::from(next_value(&mut arguments, "--dict")?)),
            "--output" => output = Some(PathBuf::from(next_value(&mut arguments, "--output")?)),
            "--rec-width" => {
                recognizer_width =
                    parse_usize(next_value(&mut arguments, "--rec-width")?, "--rec-width")?
            }
            "--det-threshold" => {
                postprocess.binary_threshold = parse_f32(
                    next_value(&mut arguments, "--det-threshold")?,
                    "--det-threshold",
                )?
            }
            "--box-threshold" => {
                postprocess.box_threshold = parse_f32(
                    next_value(&mut arguments, "--box-threshold")?,
                    "--box-threshold",
                )?
            }
            "--unclip-ratio" => {
                postprocess.unclip_ratio = parse_f32(
                    next_value(&mut arguments, "--unclip-ratio")?,
                    "--unclip-ratio",
                )?
            }
            "--min-area" => {
                postprocess.min_area =
                    parse_usize(next_value(&mut arguments, "--min-area")?, "--min-area")?
            }
            "--max-boxes" => {
                postprocess.max_boxes =
                    parse_usize(next_value(&mut arguments, "--max-boxes")?, "--max-boxes")?
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            value => bail!("unknown argument {value:?}; pass --help for usage"),
        }
    }

    if recognizer_width != embedded_width {
        bail!(
            "--rec-width {recognizer_width} does not match the recognizer ONNX width {embedded_width} embedded at build time"
        );
    }

    Ok(Arguments {
        image: image.context("--image PATH is required")?,
        dictionary: dictionary.context("--dict PATH is required")?,
        output,
        recognizer_width,
        postprocess,
    })
}

fn embedded_recognizer_width() -> usize {
    env!("PPOCR_BURN_RECOGNIZER_WIDTH")
        .parse()
        .expect("build script emits a valid recognizer width")
}

fn next_value(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<std::ffi::OsString> {
    arguments
        .next()
        .with_context(|| format!("{flag} requires a value"))
}

fn parse_f32(value: std::ffi::OsString, flag: &str) -> Result<f32> {
    value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{flag} must be valid UTF-8"))?
        .parse::<f32>()
        .with_context(|| format!("parse {flag}"))
}

fn parse_usize(value: std::ffi::OsString, flag: &str) -> Result<usize> {
    value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{flag} must be valid UTF-8"))?
        .parse::<usize>()
        .with_context(|| format!("parse {flag}"))
}

pub fn print_usage() {
    println!(
        "usage: ppocr-burn --image PATH --dict PATH [--output PATH] [--rec-width N] [--det-threshold F32] [--box-threshold F32] [--unclip-ratio F32] [--min-area N] [--max-boxes N]"
    );
}
