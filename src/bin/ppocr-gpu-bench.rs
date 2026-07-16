use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use ppocr_rs::{
    ModelKind, ModelSize, ModelStore,
    gpu::{self, Detector, Gpu, ModelOutput, Recognizer},
};
use std::{
    fs,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Debug, Parser)]
#[command(
    name = "ppocr-gpu-bench",
    version,
    about = "Benchmark one WGPU PP-OCRv6 model"
)]
struct Arguments {
    /// Model role to benchmark.
    #[arg(long, visible_alias = "model", value_enum)]
    kind: BenchmarkKind,

    /// Released model tier.
    #[arg(long, default_value_t = ModelSize::Tiny)]
    size: ModelSize,

    /// Explicit Safetensors file. Omit to use the pinned model cache.
    #[arg(long)]
    weights: Option<PathBuf>,

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

    /// Little-endian F32 input file. Omit for deterministic generated input.
    #[arg(long)]
    input: Option<PathBuf>,

    /// Compare the final output with a little-endian F32 reference file.
    #[arg(long)]
    reference: Option<PathBuf>,

    /// Write the final output as little-endian F32 values.
    #[arg(long)]
    dump: Option<PathBuf>,

    /// Number of untimed warmup runs.
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Number of timed runs.
    #[arg(long, default_value_t = 30)]
    runs: usize,
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
    fn forward(&self, input: &[f32]) -> gpu::Result<ModelOutput> {
        match self {
            Self::Detector(model) => model.forward(input),
            Self::Recognizer(model) => model.forward(input),
        }
    }

    fn benchmark(&self, input: &[f32], warmup: usize, runs: usize) -> gpu::Result<Vec<Duration>> {
        match self {
            Self::Detector(model) => model.benchmark(input, warmup, runs),
            Self::Recognizer(model) => model.benchmark(input, warmup, runs),
        }
    }
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(arguments.runs > 0, "--runs must be positive");
    let (default_height, default_width) = arguments.kind.default_dimensions();
    let height = arguments.height.unwrap_or(default_height);
    let width = arguments.width.unwrap_or(default_width);
    ensure!(height > 0, "--height must be positive");
    ensure!(width > 0, "--width must be positive");
    let input_shape = [1, 3, height, width];
    let input_length = input_shape
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .context("input shape overflow")?;
    let input = match &arguments.input {
        Some(path) => read_f32(path)?,
        None => deterministic_input(input_length),
    };
    ensure!(
        input.len() == input_length,
        "input has {} values; expected {input_length} for {input_shape:?}",
        input.len()
    );

    let weights = resolve_weights(&arguments)?;
    let gpu = Gpu::new()?;
    let model = match arguments.kind {
        BenchmarkKind::Detector => Model::Detector(Detector::load_with_size(
            &gpu,
            &weights,
            arguments.size,
            input_shape,
        )?),
        BenchmarkKind::Recognizer => Model::Recognizer(Recognizer::load_with_size(
            &gpu,
            &weights,
            arguments.size,
            input_shape,
        )?),
    };
    let output = model.forward(&input)?;
    ensure!(!output.values.is_empty(), "model output is empty");
    ensure!(
        output.values.iter().all(|value| value.is_finite()),
        "model output contains non-finite values"
    );
    let (minimum, maximum, sum) = output.values.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY, 0.0f64),
        |(minimum, maximum, sum), value| {
            (
                minimum.min(value),
                maximum.max(value),
                sum + f64::from(value),
            )
        },
    );
    let mut samples = model.benchmark(&input, arguments.warmup, arguments.runs)?;
    samples.sort_unstable();
    let average_seconds =
        samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64;

    println!("backend: gpu");
    println!("kind: {}", arguments.kind.as_str());
    println!("model: {}", weights.display());
    println!("adapter: {}", gpu.info().name);
    println!("gpu_backend: {:?}", gpu.info().backend);
    println!("gpu_device_type: {:?}", gpu.info().device_type);
    println!("input_shape: {input_shape:?}");
    println!("output_shape: {:?}", output.shape);
    println!("output_min: {minimum:.9e}");
    println!("output_max: {maximum:.9e}");
    println!("output_sum: {sum:.9e}");
    println!("warmup: {}", arguments.warmup);
    println!("runs: {}", arguments.runs);
    println!("average_ms: {:.3}", average_seconds * 1_000.0);
    println!(
        "p50_ms: {:.3}",
        percentile(&samples, 50).as_secs_f64() * 1_000.0
    );
    println!(
        "p90_ms: {:.3}",
        percentile(&samples, 90).as_secs_f64() * 1_000.0
    );
    println!(
        "p95_ms: {:.3}",
        percentile(&samples, 95).as_secs_f64() * 1_000.0
    );
    println!("throughput_per_s: {:.2}", 1.0 / average_seconds);

    if let Some(path) = &arguments.reference {
        compare(&output.values, &read_f32(path)?, &output.shape)?;
    }
    if let Some(path) = &arguments.dump {
        write_f32(path, &output.values)?;
        println!("dump: {}", path.display());
    }
    Ok(())
}

fn resolve_weights(arguments: &Arguments) -> Result<PathBuf> {
    if let Some(path) = &arguments.weights {
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

fn deterministic_input(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let value = (index.wrapping_mul(73).wrapping_add(19) % 1_024) as i32 - 512;
            value as f32 / 256.0
        })
        .collect()
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len().is_multiple_of(size_of::<f32>()),
        "{} length is not divisible by four",
        path.display()
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

fn compare(actual: &[f32], expected: &[f32], shape: &[usize]) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "reference has {} values; output has {}",
        expected.len(),
        actual.len()
    );
    let mut maximum = 0.0f32;
    let mut total = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = (actual - expected).abs();
        maximum = maximum.max(difference);
        total += f64::from(difference);
    }
    println!("reference_max_abs: {maximum:.9e}");
    println!("reference_mean_abs: {:.9e}", total / actual.len() as f64);
    if shape.len() == 3 {
        let classes = shape[2];
        let steps = actual.len() / classes;
        let matches = actual
            .chunks_exact(classes)
            .zip(expected.chunks_exact(classes))
            .filter(|(actual, expected)| argmax(actual) == argmax(expected))
            .count();
        println!("reference_argmax_matches: {matches}/{steps}");
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

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    let index = (samples.len() - 1) * percentile / 100;
    samples[index]
}
