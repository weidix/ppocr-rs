use ppocr_rs::gpu::{self, Detector, Gpu, ModelOutput, ModelSize, Recognizer};
use std::{env, error::Error, fs, io, path::PathBuf, str::FromStr, time::Duration};

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

struct Args {
    kind: String,
    size: ModelSize,
    weights: PathBuf,
    height: usize,
    width: usize,
    input: Option<PathBuf>,
    reference: Option<PathBuf>,
    warmup: usize,
    runs: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let input_shape = [1, 3, args.height, args.width];
    let input_len = input_shape
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or("input shape overflow")?;
    let input = match &args.input {
        Some(path) => read_f32(path)?,
        None => deterministic_input(input_len),
    };
    if input.len() != input_len {
        return Err(format!(
            "input has {} values; expected {input_len} for {input_shape:?}",
            input.len()
        )
        .into());
    }

    let gpu = Gpu::new()?;
    let model = match args.kind.as_str() {
        "det" | "detector" => Model::Detector(Detector::load_with_size(
            &gpu,
            &args.weights,
            args.size,
            input_shape,
        )?),
        "rec" | "recognizer" => Model::Recognizer(Recognizer::load_with_size(
            &gpu,
            &args.weights,
            args.size,
            input_shape,
        )?),
        _ => return Err("--model must be det or rec".into()),
    };
    let output = model.forward(&input)?;
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

    println!(
        "adapter={:?} backend={:?} type={:?}",
        gpu.info().name,
        gpu.info().backend,
        gpu.info().device_type
    );
    println!(
        "output_shape={:?} min={minimum:.9e} max={maximum:.9e} sum={sum:.9e}",
        output.shape
    );
    if let Some(path) = &args.reference {
        compare(&output.values, &read_f32(path)?, &output.shape)?;
    }

    let mut samples = model.benchmark(&input, args.warmup, args.runs)?;
    samples.sort_unstable();
    let p50 = percentile(&samples, 50);
    let p90 = percentile(&samples, 90);
    println!(
        "warmup={} runs={} p50_ms={:.3} p90_ms={:.3} throughput_per_s={:.2}",
        args.warmup,
        args.runs,
        p50.as_secs_f64() * 1_000.0,
        p90.as_secs_f64() * 1_000.0,
        1.0 / p50.as_secs_f64()
    );
    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut kind = None;
    let mut weights = None;
    let mut size = ModelSize::Tiny;
    let mut height = None;
    let mut width = None;
    let mut input = None;
    let mut reference = None;
    let mut warmup = 5usize;
    let mut runs = 30usize;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        if matches!(flag.as_str(), "-h" | "--help") {
            print_usage();
            std::process::exit(0);
        }
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--model" => kind = Some(value),
            "--weights" => weights = Some(PathBuf::from(value)),
            "--size" => size = ModelSize::from_str(&value)?,
            "--height" => height = Some(value.parse()?),
            "--width" => width = Some(value.parse()?),
            "--input" => input = Some(PathBuf::from(value)),
            "--reference" => reference = Some(PathBuf::from(value)),
            "--warmup" => warmup = value.parse()?,
            "--runs" => runs = value.parse()?,
            _ => return Err(format!("unknown option {flag:?}").into()),
        }
    }
    let kind = kind.ok_or("missing --model")?;
    let weights = weights.ok_or("missing --weights")?;
    let recognizer = matches!(kind.as_str(), "rec" | "recognizer");
    Ok(Args {
        kind,
        size,
        weights,
        height: height.unwrap_or(if recognizer { 48 } else { 416 }),
        width: width.unwrap_or(if recognizer { 320 } else { 736 }),
        input,
        reference,
        warmup,
        runs,
    })
}

fn print_usage() {
    println!(
        "usage: ppocr-gpu-bench --model det|rec --size medium|small|tiny \
         --weights MODEL.safetensors \
         [--height N] [--width N] [--input INPUT.f32] [--reference OUTPUT.f32] \
         [--warmup N] [--runs N]"
    );
}

fn deterministic_input(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let value = (index.wrapping_mul(73).wrapping_add(19) % 1_024) as i32 - 512;
            value as f32 / 256.0
        })
        .collect()
}

fn read_f32(path: &PathBuf) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let mut chunks = bytes.chunks_exact(4);
    let values = chunks
        .by_ref()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    if !chunks.remainder().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} length is not divisible by four", path.display()),
        )
        .into());
    }
    Ok(values)
}

fn compare(actual: &[f32], expected: &[f32], shape: &[usize]) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "reference has {} values; output has {}",
            expected.len(),
            actual.len()
        )
        .into());
    }
    let mut maximum = 0.0f32;
    let mut total = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = (actual - expected).abs();
        maximum = maximum.max(difference);
        total += f64::from(difference);
    }
    println!(
        "reference_max_abs={maximum:.9e} reference_mean_abs={:.9e}",
        total / actual.len() as f64
    );
    if shape.len() == 3 {
        let classes = shape[2];
        let steps = actual.len() / classes;
        let matches = actual
            .chunks_exact(classes)
            .zip(expected.chunks_exact(classes))
            .filter(|(actual, expected)| argmax(actual) == argmax(expected))
            .count();
        println!("reference_argmax_matches={matches}/{steps}");
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
