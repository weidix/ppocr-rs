use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use ppocr_rs::{
    model::{Detector, ModelSize, Recognizer},
    ocr::{
        DecodedText, Detection, DetectorPostprocessOptions, Point, decode_ctc_greedy_for_input,
        extract_detections, join_decoded_texts, load_dictionary, rectify_text_crop,
        split_recognition_crop_for_input,
    },
    preprocess::{load_rgb, prepare_detector_with_transform, prepare_recognizer_from_image},
};
use serde::Serialize;
use std::{env, fs, path::PathBuf};

const RECOGNIZER_HEIGHT: usize = 48;
const DEFAULT_RECOGNIZER_MAX_WIDTH: u32 = 3_200;

#[derive(Debug)]
struct Args {
    det_model: PathBuf,
    rec_model: PathBuf,
    image: PathBuf,
    dictionary: PathBuf,
    output: Option<PathBuf>,
    det_size: ModelSize,
    rec_size: ModelSize,
    det_max_side: Option<u32>,
    rec_max_width: Option<u32>,
    device: String,
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
struct Report {
    runtime: &'static str,
    device: String,
    image: String,
    source_shape: [u32; 2],
    detector_input_shape: [usize; 4],
    detector_output_shape: Vec<usize>,
    recognizer_max_width: Option<u32>,
    detector_binary_threshold: f32,
    detector_box_threshold: f32,
    detector_unclip_ratio: f32,
    texts: Vec<TextResult>,
}

struct TensorOutput {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    args.postprocess.validate()?;
    let dictionary = load_dictionary(&args.dictionary)?;
    validate_dictionary(&dictionary, args.rec_size)?;

    let device = create_device(&args.device)?;
    let image = load_rgb(&args.image)?;
    let source_shape = [image.width(), image.height()];
    let detector_input = prepare_detector_with_transform(&image, args.det_max_side)?;

    let detector = Detector::load(&args.det_model, &device, args.det_size)
        .with_context(|| format!("load detector {}", args.det_model.display()))?;
    let recognizer = Recognizer::load(&args.rec_model, &device, args.rec_size)
        .with_context(|| format!("load recognizer {}", args.rec_model.display()))?;

    let detector_output = read_output(
        "detector",
        detector.forward(&detector_input.input.to_tensor(&device)?)?,
        &device,
    )?;
    let detections = extract_detections(
        &detector_output.values,
        &detector_output.shape,
        detector_input.transform,
        args.postprocess,
    )?;

    let chunk_width = args.rec_max_width.unwrap_or(DEFAULT_RECOGNIZER_MAX_WIDTH) as usize;
    let mut texts = Vec::with_capacity(detections.len());
    for detection in detections {
        let crop = rectify_text_crop(&image, detection.polygon)?;
        let mut decoded_chunks = Vec::new();
        for chunk in split_recognition_crop_for_input(&crop, RECOGNIZER_HEIGHT, chunk_width)? {
            let recognizer_input = prepare_recognizer_from_image(&chunk, args.rec_max_width)?;
            let recognizer_output = read_output(
                "recognizer",
                recognizer.forward(&recognizer_input.input.to_tensor(&device)?)?,
                &device,
            )?;
            decoded_chunks.push(decode_ctc_greedy_for_input(
                &recognizer_output.values,
                &recognizer_output.shape,
                &dictionary,
                recognizer_input.content_width,
                recognizer_input.input.width,
            )?);
        }
        texts.push(text_result(detection, join_decoded_texts(&decoded_chunks)));
    }

    write_report(
        &Report {
            runtime: "candle-0.10.2-direct-safetensors",
            device: args.device,
            image: args.image.display().to_string(),
            source_shape,
            detector_input_shape: detector_input.input.shape(),
            detector_output_shape: detector_output.shape,
            recognizer_max_width: args.rec_max_width,
            detector_binary_threshold: args.postprocess.binary_threshold,
            detector_box_threshold: args.postprocess.box_threshold,
            detector_unclip_ratio: args.postprocess.unclip_ratio,
            texts,
        },
        args.output.as_ref(),
    )
}

fn validate_dictionary(dictionary: &[String], size: ModelSize) -> Result<()> {
    let classes = size.recognizer_classes();
    if classes != dictionary.len() + 1 && classes != dictionary.len() + 2 {
        bail!(
            "recognizer size {} has {classes} output classes, but the dictionary has {} entries",
            size.as_str(),
            dictionary.len()
        );
    }
    Ok(())
}

fn read_output(name: &str, output: Tensor, device: &Device) -> Result<TensorOutput> {
    device.synchronize()?;
    let shape = output.dims().to_vec();
    let values = output.flatten_all()?.to_vec1::<f32>()?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{name} output contains non-finite values");
    }
    Ok(TensorOutput { shape, values })
}

fn text_result(detection: Detection, decoded: DecodedText) -> TextResult {
    TextResult {
        polygon: detection.polygon,
        detection_score: detection.score,
        text: decoded.text,
        recognition_score: decoded.score,
    }
}

fn write_report(report: &Report, output: Option<&PathBuf>) -> Result<()> {
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

fn create_device(value: &str) -> Result<Device> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "metal" => Device::new_metal(0).context("create Metal device"),
        _ => bail!("unsupported --device {value:?}; expected cpu or metal"),
    }
}

fn parse_args() -> Result<Args> {
    let mut det_model = None;
    let mut rec_model = None;
    let mut image = None;
    let mut dictionary = None;
    let mut output = None;
    let mut det_size = ModelSize::Medium;
    let mut rec_size = ModelSize::Medium;
    let mut det_max_side = None;
    let mut rec_max_width = None;
    let mut device = String::from("metal");
    let mut postprocess = DetectorPostprocessOptions::default();
    let mut values = env::args().skip(1);

    while let Some(flag) = values.next() {
        if flag == "--help" || flag == "-h" {
            print_usage();
            std::process::exit(0);
        }
        let value = values
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--det-model" => det_model = Some(PathBuf::from(value)),
            "--rec-model" => rec_model = Some(PathBuf::from(value)),
            "--image" => image = Some(PathBuf::from(value)),
            "--dict" => dictionary = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--det-size" => {
                det_size = value
                    .parse()
                    .map_err(|error: String| anyhow::Error::msg(error))?
            }
            "--rec-size" => {
                rec_size = value
                    .parse()
                    .map_err(|error: String| anyhow::Error::msg(error))?
            }
            "--det-max-side" => det_max_side = Some(parse_positive_u32(&value, &flag)?),
            "--rec-max-width" => rec_max_width = Some(parse_positive_u32(&value, &flag)?),
            "--device" => device = value,
            "--det-threshold" => postprocess.binary_threshold = parse_f32(&value, &flag)?,
            "--box-threshold" => postprocess.box_threshold = parse_f32(&value, &flag)?,
            "--unclip-ratio" => postprocess.unclip_ratio = parse_f32(&value, &flag)?,
            "--min-area" => postprocess.min_area = parse_usize(&value, &flag)?,
            "--max-boxes" => postprocess.max_boxes = parse_usize(&value, &flag)?,
            _ => bail!("unknown argument {flag}; pass --help for usage"),
        }
    }

    Ok(Args {
        det_model: det_model.context("--det-model is required")?,
        rec_model: rec_model.context("--rec-model is required")?,
        image: image.context("--image is required")?,
        dictionary: dictionary.context("--dict is required")?,
        output,
        det_size,
        rec_size,
        det_max_side,
        rec_max_width,
        device,
        postprocess,
    })
}

fn parse_positive_u32(value: &str, flag: &str) -> Result<u32> {
    let parsed = value
        .parse::<u32>()
        .with_context(|| format!("parse {flag}"))?;
    if parsed == 0 {
        bail!("{flag} must be at least 1");
    }
    Ok(parsed)
}

fn parse_f32(value: &str, flag: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .with_context(|| format!("parse {flag}"))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("parse {flag}"))
}

fn print_usage() {
    println!(
        "Usage: ppocr-rs --det-model PATH --rec-model PATH --image PATH --dict PATH [--output PATH] [--det-size medium|small|tiny] [--rec-size medium|small|tiny] [--det-max-side N] [--rec-max-width N] [--device cpu|metal] [--det-threshold F32] [--box-threshold F32] [--unclip-ratio F32] [--min-area N] [--max-boxes N]"
    );
}
