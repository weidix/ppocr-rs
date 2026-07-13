use anyhow::{Context, Result, bail, ensure};
use ppocr_rs::cpu::{CpuModel, CpuOptions, Tensor};
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

struct Arguments {
    model: PathBuf,
    threads: usize,
    warmup: usize,
    runs: usize,
    dump: Option<PathBuf>,
}

fn main() -> Result<()> {
    let arguments = parse_arguments()?;
    let model = CpuModel::load(
        &arguments.model,
        CpuOptions {
            threads: arguments.threads,
        },
    )?;
    let shape = model.input_shape().to_vec();
    let length = shape.iter().product();
    let input = Tensor::from_f32(shape.clone(), deterministic_input(length))?;

    for _ in 0..arguments.warmup {
        validate_output(&model.run(input.clone())?)?;
    }
    let mut samples = Vec::with_capacity(arguments.runs);
    let mut output_shape = Vec::new();
    let mut output_sum = 0.0f64;
    let mut final_output = None;
    for run in 0..arguments.runs {
        let start = Instant::now();
        let output = model.run(input.clone())?;
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
        validate_output(&output)?;
        output_shape = output.shape().to_vec();
        output_sum = output.as_f32()?.iter().map(|&value| f64::from(value)).sum();
        if run + 1 == arguments.runs && arguments.dump.is_some() {
            final_output = Some(output);
        }
    }
    samples.sort_by(f64::total_cmp);
    println!("model: {}", arguments.model.display());
    println!("threads: {}", arguments.threads);
    println!("input: {shape:?}");
    println!("output: {output_shape:?}");
    println!("output_sum: {output_sum:.9}");
    println!("p50_ms: {:.3}", percentile(&samples, 0.50));
    println!("p90_ms: {:.3}", percentile(&samples, 0.90));
    if let (Some(path), Some(output)) = (&arguments.dump, final_output) {
        let mut writer = BufWriter::new(
            File::create(path).with_context(|| format!("create {}", path.display()))?,
        );
        for &value in output.as_f32()? {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
        println!("dump: {}", path.display());
    }
    Ok(())
}

fn parse_arguments() -> Result<Arguments> {
    let mut values = env::args().skip(1);
    let model =
        PathBuf::from(values.next().context(
            "usage: ppocr-cpu-bench MODEL.ppocr-cpu [--threads N] [--warmup N] [--runs N] [--dump OUTPUT.f32]",
        )?);
    let defaults = CpuOptions::default();
    let mut arguments = Arguments {
        model,
        threads: defaults.threads,
        warmup: 5,
        runs: 30,
        dump: None,
    };
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--dump" => arguments.dump = Some(PathBuf::from(value)),
            "--threads" => arguments.threads = parse_usize(&value, &flag)?,
            "--warmup" => arguments.warmup = parse_usize(&value, &flag)?,
            "--runs" => arguments.runs = parse_usize(&value, &flag)?,
            _ => bail!("unknown argument {flag:?}"),
        }
    }
    ensure!(arguments.threads > 0, "--threads must be positive");
    ensure!(arguments.runs > 0, "--runs must be positive");
    Ok(arguments)
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

fn validate_output(output: &Tensor) -> Result<()> {
    let values = output.as_f32()?;
    ensure!(!values.is_empty(), "model output is empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "model output contains non-finite values"
    );
    Ok(())
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}
