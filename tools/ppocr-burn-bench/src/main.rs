#![recursion_limit = "256"]

mod preprocess;

mod detector {
    include!(concat!(env!("OUT_DIR"), "/generated/detector.rs"));
}

mod recognizer {
    include!(concat!(env!("OUT_DIR"), "/generated/recognizer.rs"));
}

use anyhow::{Context, Result, bail};
use burn::{
    backend::{
        Metal,
        wgpu::{WgpuDevice, graphics, init_setup},
    },
    tensor::{Tensor, TensorData, backend::Backend},
};
use preprocess::PreparedInput;
use std::{env, hint::black_box, path::PathBuf, time::Instant};

type B = Metal;

const DETECTOR_SHAPE: [usize; 4] = preprocess::DETECTOR_SHAPE;
const RECOGNIZER_SHAPE: [usize; 4] = preprocess::RECOGNIZER_SHAPE;
const DETECTOR_OUTPUT_SHAPE: [usize; 4] = [1, 1, 416, 736];

struct Arguments {
    image: PathBuf,
    annotations: Option<PathBuf>,
    warmup: usize,
    runs: usize,
}

struct Timing {
    min_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
}

struct OutputSummary {
    shape: Vec<usize>,
    sum: f64,
}

fn main() -> Result<()> {
    let arguments = parse_arguments()?;
    let image = preprocess::load_rgb(&arguments.image)?;
    let detector_input = preprocess::prepare_fixed_detector(&image)?;
    let recognizer_input = preprocess::prepare_fixed_recognizer(
        arguments.annotations.as_deref(),
        &arguments.image,
        &image,
    )?;

    let device = WgpuDevice::default();
    init_setup::<graphics::Metal>(&device, Default::default());

    let detector = detector::Model::<B>::from_embedded(&device);
    let recognizer = recognizer::Model::<B>::from_embedded(&device);
    let detector_tensor = to_tensor(&detector_input, &device);
    let recognizer_tensor = to_tensor(&recognizer_input, &device);

    let detector_output = validate_detector_output(&detector, detector_tensor.clone(), &device)?;
    let recognizer_output =
        validate_recognizer_output(&recognizer, recognizer_tensor.clone(), &device)?;

    let detector_timing = benchmark_detector(
        &detector,
        detector_tensor,
        &device,
        arguments.warmup,
        arguments.runs,
    )?;
    let recognizer_timing = benchmark_recognizer(
        &recognizer,
        recognizer_tensor,
        &device,
        arguments.warmup,
        arguments.runs,
    )?;

    println!("backend=Burn 0.21 Metal+fusion (autotune disabled)");
    println!(
        "detector output shape={:?} sum={:.6}",
        detector_output.shape, detector_output.sum
    );
    println!(
        "recognizer output shape={:?} sum={:.6}",
        recognizer_output.shape, recognizer_output.sum
    );
    println!(
        "detector shape={DETECTOR_SHAPE:?} warmup={} runs={} min_ms={:.3} p50_ms={:.3} p90_ms={:.3}",
        arguments.warmup,
        arguments.runs,
        detector_timing.min_ms,
        detector_timing.p50_ms,
        detector_timing.p90_ms,
    );
    println!(
        "recognizer shape={RECOGNIZER_SHAPE:?} warmup={} runs={} min_ms={:.3} p50_ms={:.3} p90_ms={:.3}",
        arguments.warmup,
        arguments.runs,
        recognizer_timing.min_ms,
        recognizer_timing.p50_ms,
        recognizer_timing.p90_ms,
    );
    Ok(())
}

fn parse_arguments() -> Result<Arguments> {
    let mut image = None;
    let mut annotations = None;
    let mut warmup = 5;
    let mut runs = 30;
    let mut arguments = env::args_os().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.to_string_lossy().as_ref() {
            "--image" => image = Some(PathBuf::from(next_value(&mut arguments, "--image")?)),
            "--annotations" => {
                annotations = Some(PathBuf::from(next_value(&mut arguments, "--annotations")?))
            }
            "--warmup" => {
                warmup = parse_count(next_value(&mut arguments, "--warmup")?, "--warmup")?
            }
            "--runs" => runs = parse_count(next_value(&mut arguments, "--runs")?, "--runs")?,
            "--help" | "-h" => {
                println!(
                    "usage: ppocr-burn-bench --image PATH [--annotations JSONL] [--warmup N] [--runs N]"
                );
                std::process::exit(0);
            }
            value => bail!("unknown argument {value:?}; pass --help for usage"),
        }
    }

    if warmup == 0 {
        bail!("--warmup must be greater than zero");
    }
    if runs == 0 {
        bail!("--runs must be greater than zero");
    }
    Ok(Arguments {
        image: image.context("--image PATH is required")?,
        annotations,
        warmup,
        runs,
    })
}

fn next_value(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<std::ffi::OsString> {
    arguments
        .next()
        .with_context(|| format!("{flag} requires a value"))
}

fn parse_count(value: std::ffi::OsString, flag: &str) -> Result<usize> {
    value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{flag} must be valid UTF-8"))?
        .parse()
        .with_context(|| format!("{flag} must be an unsigned integer"))
}

fn to_tensor(input: &PreparedInput, device: &WgpuDevice) -> Tensor<B, 4> {
    Tensor::from_data(TensorData::new(input.data.clone(), input.shape()), device)
}

fn validate_detector_output(
    model: &detector::Model<B>,
    input: Tensor<B, 4>,
    device: &WgpuDevice,
) -> Result<OutputSummary> {
    let output = model.forward(input);
    let summary = read_output("detector", output, device)?;
    if summary.shape.as_slice() != DETECTOR_OUTPUT_SHAPE {
        bail!(
            "detector output shape {:?}, expected {DETECTOR_OUTPUT_SHAPE:?}",
            summary.shape
        );
    }
    Ok(summary)
}

fn validate_recognizer_output(
    model: &recognizer::Model<B>,
    input: Tensor<B, 4>,
    device: &WgpuDevice,
) -> Result<OutputSummary> {
    let output = model.forward(input);
    let summary = read_output("recognizer", output, device)?;
    if summary.shape.len() != 3
        || summary.shape[0] != 1
        || summary.shape[1] != 40
        || summary.shape[2] == 0
    {
        bail!(
            "recognizer output shape {:?}, expected [1, 40, classes>0]",
            summary.shape
        );
    }
    Ok(summary)
}

fn read_output<const D: usize>(
    name: &str,
    output: Tensor<B, D>,
    device: &WgpuDevice,
) -> Result<OutputSummary> {
    B::sync(device).with_context(|| format!("synchronize {name} output sanity check"))?;
    let data = output.into_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|error| anyhow::anyhow!("read {name} f32 output: {error}"))?;
    if values.iter().any(|value| !value.is_finite()) {
        bail!("{name} output contains non-finite values");
    }
    let sum = values.iter().map(|value| f64::from(*value)).sum::<f64>();
    if !sum.is_finite() || sum == 0.0 {
        bail!("{name} output has an invalid total sum {sum}");
    }
    Ok(OutputSummary {
        shape: data.shape.as_slice().to_vec(),
        sum,
    })
}

fn benchmark_detector(
    model: &detector::Model<B>,
    input: Tensor<B, 4>,
    device: &WgpuDevice,
    warmup: usize,
    runs: usize,
) -> Result<Timing> {
    for _ in 0..warmup {
        black_box(model.forward(input.clone()));
        B::sync(device).context("synchronize detector warmup")?;
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        black_box(model.forward(input.clone()));
        B::sync(device).context("synchronize detector")?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok(summarize(samples))
}

fn benchmark_recognizer(
    model: &recognizer::Model<B>,
    input: Tensor<B, 4>,
    device: &WgpuDevice,
    warmup: usize,
    runs: usize,
) -> Result<Timing> {
    for _ in 0..warmup {
        black_box(model.forward(input.clone()));
        B::sync(device).context("synchronize recognizer warmup")?;
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        black_box(model.forward(input.clone()));
        B::sync(device).context("synchronize recognizer")?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    Ok(summarize(samples))
}

fn summarize(mut samples: Vec<f64>) -> Timing {
    samples.sort_by(f64::total_cmp);
    let percentile = |numerator: usize, denominator: usize| {
        samples[(samples.len() - 1) * numerator / denominator]
    };
    Timing {
        min_ms: samples[0],
        p50_ms: percentile(50, 100),
        p90_ms: percentile(90, 100),
    }
}
