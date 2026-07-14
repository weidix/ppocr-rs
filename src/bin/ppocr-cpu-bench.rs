use anyhow::{Context, Result, bail, ensure};
use ppocr_rs::cpu::{CpuOptions, Detector, ModelSize, Recognizer, Tensor};
use std::{
    env, fs,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

#[derive(Clone, Copy)]
enum ModelKind {
    Detector,
    Recognizer,
}

struct Arguments {
    model: PathBuf,
    kind: ModelKind,
    size: ModelSize,
    height: usize,
    width: usize,
    threads: usize,
    warmup: usize,
    runs: usize,
    input: Option<PathBuf>,
    dump: Option<PathBuf>,
    compare: Option<PathBuf>,
}

enum Model {
    Detector(Detector),
    Recognizer(Recognizer),
}

impl Model {
    fn run(&self, input: Tensor) -> Result<Tensor> {
        match self {
            Self::Detector(model) => model.run(input),
            Self::Recognizer(model) => model.run(input),
        }
    }
}

fn main() -> Result<()> {
    let arguments = parse_arguments()?;
    let options = CpuOptions {
        threads: arguments.threads,
    };
    let model = match arguments.kind {
        ModelKind::Detector => {
            Model::Detector(Detector::load(&arguments.model, arguments.size, options)?)
        }
        ModelKind::Recognizer => {
            Model::Recognizer(Recognizer::load(&arguments.model, arguments.size, options)?)
        }
    };
    let shape = vec![1, 3, arguments.height, arguments.width];
    let length = shape.iter().product();
    let values = match &arguments.input {
        Some(path) => read_f32(path, length)?,
        None => deterministic_input(length),
    };
    let input = Tensor::from_f32(shape.clone(), values)?;

    for _ in 0..arguments.warmup {
        validate_output(&model.run(input.clone())?)?;
    }
    let mut samples = Vec::with_capacity(arguments.runs);
    let mut final_output = None;
    for run in 0..arguments.runs {
        let start = Instant::now();
        let output = model.run(input.clone())?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
        validate_output(&output)?;
        if run + 1 == arguments.runs {
            final_output = Some(output);
        }
    }
    samples.sort_by(f64::total_cmp);
    let output = final_output.expect("positive run count");
    let values = output.as_f32()?;
    let output_sum = values.iter().map(|&value| f64::from(value)).sum::<f64>();
    println!("model: {}", arguments.model.display());
    println!("threads: {}", arguments.threads);
    println!("input: {shape:?}");
    println!("output: {:?}", output.shape());
    println!("output_sum: {output_sum:.9}");
    println!("p50_ms: {:.3}", percentile(&samples, 0.50));
    println!("p90_ms: {:.3}", percentile(&samples, 0.90));

    if let Some(path) = &arguments.compare {
        let reference = read_f32(path, values.len())?;
        report_difference(arguments.kind, output.shape(), values, &reference)?;
    }
    if let Some(path) = &arguments.dump {
        write_f32(path, values)?;
        println!("dump: {}", path.display());
    }
    Ok(())
}

fn parse_arguments() -> Result<Arguments> {
    let mut values = env::args().skip(1);
    let mut model = None;
    let mut kind = None;
    let mut size = ModelSize::Tiny;
    let mut height = None;
    let mut width = None;
    let defaults = CpuOptions::default();
    let mut threads = defaults.threads;
    let mut warmup = 5;
    let mut runs = 30;
    let mut input = None;
    let mut dump = None;
    let mut compare = None;
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--kind" => {
                kind = Some(match value.as_str() {
                    "det" => ModelKind::Detector,
                    "rec" => ModelKind::Recognizer,
                    _ => bail!("unsupported --kind {value:?}; expected det or rec"),
                })
            }
            "--size" => size = ModelSize::from_str(&value).map_err(anyhow::Error::msg)?,
            "--height" => height = Some(parse_usize(&value, &flag)?),
            "--width" => width = Some(parse_usize(&value, &flag)?),
            "--threads" => threads = parse_usize(&value, &flag)?,
            "--warmup" => warmup = parse_usize(&value, &flag)?,
            "--runs" => runs = parse_usize(&value, &flag)?,
            "--input" => input = Some(PathBuf::from(value)),
            "--dump" => dump = Some(PathBuf::from(value)),
            "--compare" => compare = Some(PathBuf::from(value)),
            _ => bail!("unknown argument {flag:?}"),
        }
    }
    let model = model.context(
        "usage: ppocr-cpu-bench --model MODEL.safetensors --kind det|rec [--size medium|small|tiny] [--height N] [--width N] [--threads N] [--warmup N] [--runs N] [--input INPUT.f32] [--dump OUTPUT.f32] [--compare REFERENCE.f32]",
    )?;
    let kind = kind.context("--kind is required")?;
    let (default_height, default_width) = match kind {
        ModelKind::Detector => (416, 736),
        ModelKind::Recognizer => (48, 320),
    };
    ensure!(threads > 0, "--threads must be positive");
    ensure!(runs > 0, "--runs must be positive");
    Ok(Arguments {
        model,
        kind,
        size,
        height: height.unwrap_or(default_height),
        width: width.unwrap_or(default_width),
        threads,
        warmup,
        runs,
        input,
        dump,
        compare,
    })
}

fn parse_usize(value: &str, flag: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("parse {flag}"))
}

fn deterministic_input(length: usize) -> Vec<f32> {
    let mut state = 0x243f_6a88u32;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 0x00ff_ffff as f32) * 2.0 - 1.0
        })
        .collect()
}

fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() == expected * size_of::<f32>(),
        "{} must contain {expected} F32 values, found {} bytes",
        path.display(),
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect())
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn validate_output(output: &Tensor) -> Result<()> {
    let values = output.as_f32()?;
    ensure!(!values.is_empty(), "model output is empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "model output contains non-finite values"
    );
    Ok(())
}

fn report_difference(
    kind: ModelKind,
    shape: &[usize],
    actual: &[f32],
    expected: &[f32],
) -> Result<()> {
    let mut max_absolute = 0.0f32;
    let mut total_absolute = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = (actual - expected).abs();
        max_absolute = max_absolute.max(difference);
        total_absolute += f64::from(difference);
    }
    println!("max_abs_error: {max_absolute:.9}");
    println!(
        "mean_abs_error: {:.9}",
        total_absolute / actual.len() as f64
    );
    match kind {
        ModelKind::Detector => {
            let mismatches = actual
                .iter()
                .zip(expected)
                .filter(|&(actual, expected)| (*actual >= 0.5) != (*expected >= 0.5))
                .count();
            println!("threshold_0.5_mismatches: {mismatches}");
        }
        ModelKind::Recognizer => {
            let classes = *shape
                .last()
                .context("recognizer output has no class axis")?;
            ensure!(classes > 0, "recognizer class axis is empty");
            let mismatches = actual
                .chunks_exact(classes)
                .zip(expected.chunks_exact(classes))
                .filter(|(actual, expected)| argmax(actual) != argmax(expected))
                .count();
            println!("argmax_mismatches: {mismatches}");
        }
    }
    Ok(())
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}
