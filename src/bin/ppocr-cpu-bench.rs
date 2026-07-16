mod benchmark;
mod model_store;

use anyhow::{Context, Result, ensure};
use benchmark::{
    BenchmarkArgs, BenchmarkKind, print_benchmark_report, read_f32, report_reference,
    resolve_model_path, summarize_output, validate_output, write_f32,
};
use clap::Parser;
use ppocr_rs::{
    CpuOptions,
    cpu::{Detector, Recognizer, Tensor},
};
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(
    name = "ppocr-cpu-bench",
    version,
    about = "Benchmark one native CPU PP-OCRv6 model"
)]
struct Arguments {
    #[command(flatten)]
    benchmark: BenchmarkArgs,

    /// Number of CPU worker threads.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
}

enum Model {
    Detector(Detector),
    Recognizer(Recognizer),
}

impl Model {
    fn forward(&self, input: Tensor) -> Result<Tensor> {
        match self {
            Self::Detector(model) => model.forward(input),
            Self::Recognizer(model) => model.forward(input),
        }
    }
}

fn main() -> Result<()> {
    let Arguments { benchmark, threads } = Arguments::parse();
    ensure!(threads > 0, "--threads must be positive");

    let input_shape = benchmark.input_shape()?;
    let values = benchmark.input_values(input_shape)?;
    let input = Tensor::from_f32(input_shape, values)?;
    let model_path = resolve_model_path(&benchmark)?;
    let options = CpuOptions { threads };
    let model = match benchmark.kind {
        BenchmarkKind::Detector => {
            Model::Detector(Detector::load(&model_path, benchmark.model_size, options)?)
        }
        BenchmarkKind::Recognizer => Model::Recognizer(Recognizer::load(
            &model_path,
            benchmark.model_size,
            options,
        )?),
    };

    for _ in 0..benchmark.warmup {
        let output = model.forward(input.clone())?;
        validate_output(output.as_f32()?)?;
    }
    let mut samples = Vec::with_capacity(benchmark.runs);
    let mut final_output = None;
    for _ in 0..benchmark.runs {
        let start = Instant::now();
        let output = model.forward(input.clone())?;
        samples.push(start.elapsed());
        validate_output(output.as_f32()?)?;
        final_output = Some(output);
    }

    let output = final_output.context("benchmark did not produce an output")?;
    let values = output.as_f32()?;
    let summary = summarize_output(values)?;
    print_benchmark_report(
        "cpu",
        &benchmark,
        &model_path,
        input_shape,
        output.shape(),
        summary,
        &samples,
    )?;

    if let Some(path) = &benchmark.reference {
        let reference = read_f32(path, values.len())?;
        report_reference(benchmark.kind, output.shape(), values, &reference)?;
    }
    if let Some(path) = &benchmark.dump {
        write_f32(path, values)?;
        println!("dump: {}", path.display());
    }
    Ok(())
}

fn default_threads() -> usize {
    CpuOptions::default().threads
}
