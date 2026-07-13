use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use ppocr_rs::model::{Detector, ModelSize, Recognizer};
use ppocr_rs::preprocess::{
    Crop, PreparedInput, load_rgb, longest_annotation_crop, prepare_detector, prepare_recognizer,
};
use serde::Serialize;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
struct Args {
    det_model: PathBuf,
    rec_model: PathBuf,
    image: PathBuf,
    annotations: PathBuf,
    det_size: ModelSize,
    rec_size: ModelSize,
    det_max_side: Option<u32>,
    rec_max_width: Option<u32>,
    device: String,
    warmup: usize,
    iterations: usize,
}

#[derive(Serialize)]
struct Latency {
    min_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    mean_ms: f64,
    max_ms: f64,
}

#[derive(Serialize)]
struct ModelReport {
    model_load_ms: f64,
    preprocessing_ms: f64,
    h2d_ms: Latency,
    forward_ms: Latency,
    h2d_and_forward_ms: Latency,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    output_sum: f32,
}

#[derive(Serialize)]
struct Report {
    runtime: &'static str,
    device: String,
    image: String,
    source_shape: [u32; 2],
    detector_size: &'static str,
    recognizer_size: &'static str,
    detector_max_side: Option<u32>,
    recognizer_max_width: Option<u32>,
    jpeg_decode_ms: f64,
    warmup: usize,
    iterations: usize,
    recognition_crop: Crop,
    detector: ModelReport,
    recognizer: ModelReport,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let device = create_device(&args.device)?;

    let decode_started = Instant::now();
    let image = load_rgb(&args.image)?;
    let jpeg_decode_ms = elapsed_ms(decode_started);
    let source_shape = [image.width(), image.height()];

    let detector_preprocess_started = Instant::now();
    let detector_input = prepare_detector(&image, args.det_max_side);
    let detector_preprocessing_ms = elapsed_ms(detector_preprocess_started);

    let crop = longest_annotation_crop(&args.annotations, &args.image, &image)?;
    let recognizer_preprocess_started = Instant::now();
    let recognizer_input = prepare_recognizer(&image, crop, args.rec_max_width)?;
    let recognizer_preprocessing_ms = elapsed_ms(recognizer_preprocess_started);

    let detector_load_started = Instant::now();
    let detector = Detector::load(&args.det_model, &device, args.det_size)
        .with_context(|| format!("load detector {}", args.det_model.display()))?;
    let detector_load_ms = elapsed_ms(detector_load_started);
    let detector_report = benchmark_detector(
        &detector,
        &detector_input,
        &device,
        args.warmup,
        args.iterations,
        detector_load_ms,
        detector_preprocessing_ms,
    )?;

    let recognizer_load_started = Instant::now();
    let recognizer = Recognizer::load(&args.rec_model, &device, args.rec_size)
        .with_context(|| format!("load recognizer {}", args.rec_model.display()))?;
    let recognizer_load_ms = elapsed_ms(recognizer_load_started);
    let recognizer_report = benchmark_recognizer(
        &recognizer,
        &recognizer_input,
        &device,
        args.warmup,
        args.iterations,
        recognizer_load_ms,
        recognizer_preprocessing_ms,
    )?;

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            runtime: "candle-0.10.2-direct-safetensors",
            device: args.device,
            image: args.image.display().to_string(),
            source_shape,
            detector_size: args.det_size.as_str(),
            recognizer_size: args.rec_size.as_str(),
            detector_max_side: args.det_max_side,
            recognizer_max_width: args.rec_max_width,
            jpeg_decode_ms,
            warmup: args.warmup,
            iterations: args.iterations,
            recognition_crop: crop,
            detector: detector_report,
            recognizer: recognizer_report,
        })?
    );
    Ok(())
}

fn benchmark_detector(
    model: &Detector,
    input: &PreparedInput,
    device: &Device,
    warmup: usize,
    iterations: usize,
    model_load_ms: f64,
    preprocessing_ms: f64,
) -> Result<ModelReport> {
    let h2d_ms = measure_h2d(input, device, iterations)?;
    let device_input = input.to_tensor(device)?;
    for _ in 0..warmup {
        let _ = model.forward(&device_input)?;
        device.synchronize()?;
    }
    let (forward_ms, output) =
        measure_forward(device, iterations, || model.forward(&device_input))?;
    let h2d_and_forward_ms =
        measure_h2d_and_forward(device, input, iterations, |tensor| model.forward(tensor))?;
    report_from_output(
        model_load_ms,
        preprocessing_ms,
        input,
        output,
        h2d_ms,
        forward_ms,
        h2d_and_forward_ms,
    )
}

fn benchmark_recognizer(
    model: &Recognizer,
    input: &PreparedInput,
    device: &Device,
    warmup: usize,
    iterations: usize,
    model_load_ms: f64,
    preprocessing_ms: f64,
) -> Result<ModelReport> {
    let h2d_ms = measure_h2d(input, device, iterations)?;
    let device_input = input.to_tensor(device)?;
    for _ in 0..warmup {
        let _ = model.forward(&device_input)?;
        device.synchronize()?;
    }
    let (forward_ms, output) =
        measure_forward(device, iterations, || model.forward(&device_input))?;
    let h2d_and_forward_ms =
        measure_h2d_and_forward(device, input, iterations, |tensor| model.forward(tensor))?;
    report_from_output(
        model_load_ms,
        preprocessing_ms,
        input,
        output,
        h2d_ms,
        forward_ms,
        h2d_and_forward_ms,
    )
}

fn measure_h2d(input: &PreparedInput, device: &Device, iterations: usize) -> Result<Latency> {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let _ = input.to_tensor(device)?;
        device.synchronize()?;
        samples.push(elapsed_ms(started));
    }
    Ok(latency(samples))
}

fn measure_forward<F>(
    device: &Device,
    iterations: usize,
    mut forward: F,
) -> Result<(Latency, Tensor)>
where
    F: FnMut() -> candle_core::Result<Tensor> + Send,
{
    let mut samples = Vec::with_capacity(iterations);
    let mut output = None;
    for _ in 0..iterations {
        let started = Instant::now();
        let result = forward()?;
        device.synchronize()?;
        samples.push(elapsed_ms(started));
        output = Some(result);
    }
    Ok((latency(samples), output.expect("at least one iteration")))
}

fn measure_h2d_and_forward<F>(
    device: &Device,
    input: &PreparedInput,
    iterations: usize,
    mut forward: F,
) -> Result<Latency>
where
    F: FnMut(&Tensor) -> candle_core::Result<Tensor> + Send,
{
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let tensor = input.to_tensor(device)?;
        let _ = forward(&tensor)?;
        device.synchronize()?;
        samples.push(elapsed_ms(started));
    }
    Ok(latency(samples))
}

fn report_from_output(
    model_load_ms: f64,
    preprocessing_ms: f64,
    input: &PreparedInput,
    output: Tensor,
    h2d_ms: Latency,
    forward_ms: Latency,
    h2d_and_forward_ms: Latency,
) -> Result<ModelReport> {
    let output_shape = output.dims().to_vec();
    let output_sum = output.sum_all()?.to_scalar::<f32>()?;
    Ok(ModelReport {
        model_load_ms,
        preprocessing_ms,
        h2d_ms,
        forward_ms,
        h2d_and_forward_ms,
        input_shape: input.shape().to_vec(),
        output_shape,
        output_sum,
    })
}

fn latency(mut samples: Vec<f64>) -> Latency {
    samples.sort_by(f64::total_cmp);
    let count = samples.len();
    let percentile = |p: f64| samples[((count - 1) as f64 * p).round() as usize];
    Latency {
        min_ms: samples[0],
        p50_ms: percentile(0.50),
        p90_ms: percentile(0.90),
        mean_ms: samples.iter().sum::<f64>() / count as f64,
        max_ms: samples[count - 1],
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
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
    let mut annotations = None;
    let mut det_size = ModelSize::Medium;
    let mut rec_size = ModelSize::Medium;
    let mut det_max_side = None;
    let mut rec_max_width = None;
    let mut device = String::from("metal");
    let mut warmup = 2;
    let mut iterations = 5;
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
            "--annotations" => annotations = Some(PathBuf::from(value)),
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
            "--warmup" => warmup = value.parse().context("parse --warmup")?,
            "--iterations" => iterations = value.parse().context("parse --iterations")?,
            _ => bail!("unknown argument {flag}"),
        }
    }
    if iterations == 0 {
        bail!("--iterations must be at least 1");
    }
    Ok(Args {
        det_model: det_model.context("--det-model is required")?,
        rec_model: rec_model.context("--rec-model is required")?,
        image: image.context("--image is required")?,
        annotations: annotations.context("--annotations is required")?,
        det_size,
        rec_size,
        det_max_side,
        rec_max_width,
        device,
        warmup,
        iterations,
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

fn print_usage() {
    println!(
        "Usage: ppocr-rs --det-model PATH --rec-model PATH --image PATH --annotations PATH [--det-size medium|small|tiny] [--rec-size medium|small|tiny] [--det-max-side N] [--rec-max-width N] [--device cpu|metal] [--warmup N] [--iterations N]"
    );
}
