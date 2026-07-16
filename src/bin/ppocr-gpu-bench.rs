mod benchmark;
mod model_store;

use anyhow::Result;
use benchmark::{
    BenchmarkArgs, BenchmarkKind, print_benchmark_report, read_f32, report_reference,
    resolve_model_path, summarize_output, write_f32,
};
use clap::Parser;
use ppocr_rs::gpu::{self, Detector, Gpu, ModelOutput, Recognizer};
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(
    name = "ppocr-gpu-bench",
    version,
    about = "Benchmark one WGPU PP-OCRv6 model"
)]
struct Arguments {
    #[command(flatten)]
    benchmark: BenchmarkArgs,
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
    let Arguments { benchmark } = Arguments::parse();
    let input_shape = benchmark.input_shape()?;
    let input = benchmark.input_values(input_shape)?;
    let model_path = resolve_model_path(&benchmark)?;
    let gpu = Gpu::new()?;
    let model = match benchmark.kind {
        BenchmarkKind::Detector => Model::Detector(Detector::load(
            &gpu,
            &model_path,
            benchmark.model_size,
            input_shape,
        )?),
        BenchmarkKind::Recognizer => Model::Recognizer(Recognizer::load(
            &gpu,
            &model_path,
            benchmark.model_size,
            input_shape,
        )?),
    };

    let output = model.forward(&input)?;
    let summary = summarize_output(&output.values)?;
    let samples = model.benchmark(&input, benchmark.warmup, benchmark.runs)?;
    print_benchmark_report(
        "gpu",
        &benchmark,
        &model_path,
        input_shape,
        &output.shape,
        summary,
        &samples,
    )?;

    if let Some(path) = &benchmark.reference {
        let reference = read_f32(path, output.values.len())?;
        report_reference(benchmark.kind, &output.shape, &output.values, &reference)?;
    }
    if let Some(path) = &benchmark.dump {
        write_f32(path, &output.values)?;
        println!("dump: {}", path.display());
    }
    Ok(())
}
