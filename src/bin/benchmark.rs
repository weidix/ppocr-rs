use super::model_store::ModelStoreArgs;
use anyhow::{Context, Result, ensure};
use clap::{Args, ValueEnum};
use ppocr_rs::{ModelKind, ModelSize, ModelStore};
use std::{
    fs,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    /// Model role to benchmark.
    #[arg(long, value_enum)]
    pub kind: BenchmarkKind,

    /// Released model tier.
    #[arg(long, default_value_t = ModelSize::Tiny)]
    pub model_size: ModelSize,

    /// Explicit Safetensors file. Omit to use the pinned model cache.
    #[arg(long)]
    pub model: Option<PathBuf>,

    #[command(flatten)]
    pub model_store: ModelStoreArgs,

    /// Input height. Defaults to 416 for detector and 48 for recognizer.
    #[arg(long)]
    pub height: Option<usize>,

    /// Input width. Defaults to 736 for detector and 320 for recognizer.
    #[arg(long)]
    pub width: Option<usize>,

    /// Little-endian F32 input file. Omit for deterministic generated input.
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Compare the final output with a little-endian F32 reference file.
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Write the final output as little-endian F32 values.
    #[arg(long)]
    pub dump: Option<PathBuf>,

    /// Number of untimed warmup runs.
    #[arg(long, default_value_t = 5)]
    pub warmup: usize,

    /// Number of timed runs.
    #[arg(long, default_value_t = 30)]
    pub runs: usize,
}

impl BenchmarkArgs {
    pub fn input_shape(&self) -> Result<[usize; 4]> {
        ensure!(self.runs > 0, "--runs must be positive");
        let (default_height, default_width) = self.kind.default_dimensions();
        let height = self.height.unwrap_or(default_height);
        let width = self.width.unwrap_or(default_width);
        ensure!(height > 0, "--height must be positive");
        ensure!(width > 0, "--width must be positive");
        Ok([1, 3, height, width])
    }

    pub fn input_values(&self, shape: [usize; 4]) -> Result<Vec<f32>> {
        let length = shape
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .context("input shape overflow")?;
        match &self.input {
            Some(path) => read_f32(path, length),
            None => Ok(deterministic_input(length)),
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BenchmarkKind {
    #[value(name = "det")]
    Detector,
    #[value(name = "rec")]
    Recognizer,
}

impl BenchmarkKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Detector => "det",
            Self::Recognizer => "rec",
        }
    }

    pub const fn model_kind(self) -> ModelKind {
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

pub fn resolve_model_path(arguments: &BenchmarkArgs) -> Result<PathBuf> {
    if let Some(path) = &arguments.model {
        return Ok(path.clone());
    }
    let store = ModelStore::new(&arguments.model_store.model_dir);
    let paths = store.resolve(
        arguments.kind.model_kind(),
        arguments.model_size,
        arguments.model_store.access(),
    )?;
    Ok(paths.weights)
}

pub fn deterministic_input(length: usize) -> Vec<f32> {
    let mut state = 0x243f_6a88u32;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 0x00ff_ffff as f32) * 2.0 - 1.0
        })
        .collect()
}

pub fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>> {
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

pub fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let mut writer =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

pub fn validate_output(values: &[f32]) -> Result<()> {
    ensure!(!values.is_empty(), "model output is empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "model output contains non-finite values"
    );
    Ok(())
}

pub struct OutputSummary {
    minimum: f32,
    maximum: f32,
    sum: f64,
}

pub fn summarize_output(values: &[f32]) -> Result<OutputSummary> {
    validate_output(values)?;
    let (minimum, maximum, sum) = values.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY, 0.0f64),
        |(minimum, maximum, sum), value| {
            (
                minimum.min(value),
                maximum.max(value),
                sum + f64::from(value),
            )
        },
    );
    Ok(OutputSummary {
        minimum,
        maximum,
        sum,
    })
}

pub fn print_benchmark_report(
    backend: &str,
    arguments: &BenchmarkArgs,
    model_path: &Path,
    input_shape: [usize; 4],
    output_shape: &[usize],
    output: OutputSummary,
    samples: &[Duration],
) -> Result<()> {
    ensure!(
        samples.len() == arguments.runs,
        "benchmark produced {} samples; expected {}",
        samples.len(),
        arguments.runs
    );
    let statistics = timing_statistics(samples)?;

    println!("backend: {backend}");
    println!("kind: {}", arguments.kind.as_str());
    println!("model: {}", model_path.display());
    println!("input_shape: {input_shape:?}");
    println!("output_shape: {output_shape:?}");
    println!("output_min: {:.9e}", output.minimum);
    println!("output_max: {:.9e}", output.maximum);
    println!("output_sum: {:.9e}", output.sum);
    println!("warmup: {}", arguments.warmup);
    println!("runs: {}", arguments.runs);
    println!("average_ms: {:.3}", statistics.average_ms);
    println!("p50_ms: {:.3}", statistics.p50_ms);
    println!("p90_ms: {:.3}", statistics.p90_ms);
    println!("p95_ms: {:.3}", statistics.p95_ms);
    println!("throughput: {:.2}", statistics.throughput_per_second);
    Ok(())
}

pub fn report_reference(
    kind: BenchmarkKind,
    shape: &[usize],
    actual: &[f32],
    expected: &[f32],
) -> Result<()> {
    validate_output(actual)?;
    ensure!(
        actual.len() == expected.len(),
        "reference has {} values; output has {}",
        expected.len(),
        actual.len()
    );
    ensure!(
        expected.iter().all(|value| value.is_finite()),
        "reference contains non-finite values"
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
            ensure!(
                actual.len().is_multiple_of(classes),
                "recognizer output length {} is not divisible by class count {classes}",
                actual.len()
            );
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

struct TimingStatistics {
    average_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    throughput_per_second: f64,
}

fn timing_statistics(samples: &[Duration]) -> Result<TimingStatistics> {
    ensure!(!samples.is_empty(), "benchmark produced no samples");
    let average_ms = samples
        .iter()
        .map(|sample| sample.as_secs_f64())
        .sum::<f64>()
        / samples.len() as f64
        * 1_000.0;
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    Ok(TimingStatistics {
        average_ms,
        p50_ms: percentile(&sorted, 0.50).as_secs_f64() * 1_000.0,
        p90_ms: percentile(&sorted, 0.90).as_secs_f64() * 1_000.0,
        p95_ms: percentile(&sorted, 0.95).as_secs_f64() * 1_000.0,
        throughput_per_second: 1_000.0 / average_ms,
    })
}

fn percentile(sorted: &[Duration], percentile: f64) -> Duration {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}
