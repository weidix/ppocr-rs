use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use ppocr_rs::{
    CpuOptions, ModelKind, ModelSize, ModelStore,
    cpu::{Detector, Recognizer, Tensor},
};
use std::{
    fs,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Parser)]
#[command(
    name = "ppocr-cpu-bench",
    version,
    about = "Benchmark one native CPU PP-OCRv6 model"
)]
struct Arguments {
    /// Model role to benchmark.
    #[arg(long, value_enum)]
    kind: BenchmarkKind,

    /// Released model tier.
    #[arg(long, default_value_t = ModelSize::Tiny)]
    size: ModelSize,

    /// Explicit Safetensors file. Omit to use the pinned model cache.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Directory where pinned model packages are stored.
    #[arg(long, env = "PPOCR_MODEL_DIR", default_value = "models")]
    model_dir: PathBuf,

    /// Fail when the pinned model is not already cached.
    #[arg(long)]
    offline: bool,

    /// Recompute hashes for the pinned model package before loading it.
    #[arg(long)]
    verify_models: bool,

    /// Input height. Defaults to 416 for detector and 48 for recognizer.
    #[arg(long)]
    height: Option<usize>,

    /// Input width. Defaults to 736 for detector and 320 for recognizer.
    #[arg(long)]
    width: Option<usize>,

    /// Number of CPU worker threads.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,

    /// Number of untimed warmup runs.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Number of timed runs.
    #[arg(long, default_value_t = 30)]
    runs: usize,

    /// Little-endian F32 input file. Omit for deterministic generated input.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Write the final output as little-endian F32 values.
    #[arg(long)]
    dump: Option<PathBuf>,

    /// Compare the final output with a little-endian F32 reference file.
    #[arg(long, visible_alias = "compare")]
    reference: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchmarkKind {
    #[value(name = "det")]
    Detector,
    #[value(name = "rec")]
    Recognizer,
}

impl BenchmarkKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Detector => "det",
            Self::Recognizer => "rec",
        }
    }

    const fn model_kind(self) -> ModelKind {
        match self {
            Self::Detector => ModelKind::Detector,
            Self::Recognizer => ModelKind::Recognizer,
        }
    }

    const fn default_dimensions(self) -> (usize, usize) {
        match self {
            Self::Detector => (416, 736),
            Self::Recognizer => (48, 320),
        }
    }
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
    let arguments = Arguments::parse();
    ensure!(arguments.threads > 0, "--threads must be positive");
    ensure!(arguments.runs > 0, "--runs must be positive");
    let (default_height, default_width) = arguments.kind.default_dimensions();
    let height = arguments.height.unwrap_or(default_height);
    let width = arguments.width.unwrap_or(default_width);
    ensure!(height > 0, "--height must be positive");
    ensure!(width > 0, "--width must be positive");

    let model_path = resolve_model_path(&arguments)?;
    let options = CpuOptions {
        threads: arguments.threads,
    };
    let model = match arguments.kind {
        BenchmarkKind::Detector => {
            Model::Detector(Detector::load(&model_path, arguments.size, options)?)
        }
        BenchmarkKind::Recognizer => {
            Model::Recognizer(Recognizer::load(&model_path, arguments.size, options)?)
        }
    };
    let input_shape = [1, 3, height, width];
    let input_length = input_shape
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .context("input shape overflow")?;
    let values = match &arguments.input {
        Some(path) => read_f32(path, input_length)?,
        None => deterministic_input(input_length),
    };
    let input = Tensor::from_f32(input_shape, values)?;

    for _ in 0..arguments.warmup {
        validate_output(&model.run(input.clone())?)?;
    }
    let mut samples = Vec::with_capacity(arguments.runs);
    let mut final_output = None;
    for _ in 0..arguments.runs {
        let start = Instant::now();
        let output = model.run(input.clone())?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
        validate_output(&output)?;
        final_output = Some(output);
    }
    let output = final_output.context("benchmark did not produce an output")?;
    let values = output.as_f32()?;
    let output_sum = values.iter().map(|&value| f64::from(value)).sum::<f64>();
    let average_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    samples.sort_by(f64::total_cmp);

    println!("backend: cpu");
    println!("kind: {}", arguments.kind.as_str());
    println!("model: {}", model_path.display());
    println!("threads: {}", arguments.threads);
    println!("input_shape: {input_shape:?}");
    println!("output_shape: {:?}", output.shape());
    println!("output_sum: {output_sum:.9}");
    println!("warmup: {}", arguments.warmup);
    println!("runs: {}", arguments.runs);
    println!("average_ms: {average_ms:.3}");
    println!("p50_ms: {:.3}", percentile(&samples, 0.50));
    println!("p90_ms: {:.3}", percentile(&samples, 0.90));
    println!("p95_ms: {:.3}", percentile(&samples, 0.95));
    println!("throughput_per_s: {:.2}", 1_000.0 / average_ms);

    if let Some(path) = &arguments.reference {
        let reference = read_f32(path, values.len())?;
        report_difference(arguments.kind, output.shape(), values, &reference)?;
    }
    if let Some(path) = &arguments.dump {
        write_f32(path, values)?;
        println!("dump: {}", path.display());
    }
    Ok(())
}

fn resolve_model_path(arguments: &Arguments) -> Result<PathBuf> {
    if let Some(path) = &arguments.model {
        return Ok(path.clone());
    }
    let store = ModelStore::new(&arguments.model_dir);
    let kind = arguments.kind.model_kind();
    let paths = if arguments.verify_models {
        store.verify(kind, arguments.size)?
    } else if arguments.offline {
        store.ensure_offline(kind, arguments.size)?
    } else {
        store.ensure(kind, arguments.size)?
    };
    Ok(paths.weights)
}

fn default_threads() -> usize {
    CpuOptions::default().threads
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
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
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
    kind: BenchmarkKind,
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
    println!("reference_max_abs: {max_absolute:.9}");
    println!(
        "reference_mean_abs: {:.9}",
        total_absolute / actual.len() as f64
    );
    match kind {
        BenchmarkKind::Detector => {
            let mismatches = actual
                .iter()
                .zip(expected)
                .filter(|&(actual, expected)| (*actual >= 0.5) != (*expected >= 0.5))
                .count();
            println!("reference_threshold_0.5_mismatches: {mismatches}");
        }
        BenchmarkKind::Recognizer => {
            let classes = *shape
                .last()
                .context("recognizer output has no class axis")?;
            ensure!(classes > 0, "recognizer class axis is empty");
            let mismatches = actual
                .chunks_exact(classes)
                .zip(expected.chunks_exact(classes))
                .filter(|(actual, expected)| argmax(actual) != argmax(expected))
                .count();
            println!("reference_argmax_mismatches: {mismatches}");
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
