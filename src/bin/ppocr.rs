mod model_store;

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use model_store::ModelStoreArgs;
use ppocr_rs::{
    DetectorPostprocessOptions, ModelKind, ModelPaths, ModelSize, ModelStore, OcrBackend,
    OcrEngine, OcrOptions,
};
use std::{fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(name = "ppocr", version, about = "Run end-to-end PP-OCRv6 inference")]
struct Arguments {
    /// JPEG or PNG image to recognize.
    #[arg(value_name = "IMAGE")]
    image: PathBuf,

    /// Inference backend.
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,

    /// Released model tier used for both detector and recognizer.
    #[arg(long, default_value_t = ModelSize::Tiny)]
    model_size: ModelSize,

    #[command(flatten)]
    model_store: ModelStoreArgs,

    /// Use an explicit detector Safetensors file instead of the pinned package.
    #[arg(long)]
    detector_model: Option<PathBuf>,

    /// Use an explicit recognizer Safetensors file instead of the pinned package.
    #[arg(long)]
    recognizer_model: Option<PathBuf>,

    /// Use an explicit recognizer dictionary or inference.yml file.
    #[arg(long)]
    dictionary: Option<PathBuf>,

    /// Number of CPU worker threads. The GPU backend does not use this option.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,

    /// Resize the detector input so its longest side is no larger than this value.
    #[arg(long)]
    detector_max_side: Option<u32>,

    /// Recognizer canvas width and maximum width of one recognition chunk.
    #[arg(long, default_value_t = 320)]
    recognizer_max_width: u32,

    /// Minimum detector probability used to form a text component.
    #[arg(long, default_value_t = 0.2)]
    binary_threshold: f32,

    /// Minimum average detector probability required to keep a text region.
    #[arg(long, default_value_t = 0.4)]
    box_threshold: f32,

    /// Minimum number of detector pixels required to keep a text region.
    #[arg(long, default_value_t = 3)]
    min_area: usize,

    /// Scale applied while expanding detector regions before recognition.
    #[arg(long, default_value_t = 1.4)]
    unclip_ratio: f32,

    /// Maximum number of detected text regions to recognize.
    #[arg(long, default_value_t = 1_000)]
    max_boxes: usize,

    /// Result representation written to stdout or --output.
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,

    /// Write the result to a file instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}

impl From<Backend> for OcrBackend {
    fn from(value: Backend) -> Self {
        match value {
            Backend::Cpu => Self::Cpu,
            Backend::Gpu => Self::Gpu,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Json,
    Text,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(arguments.threads > 0, "--threads must be positive");
    let options = OcrOptions {
        backend: arguments.backend.into(),
        detector_size: arguments.model_size,
        recognizer_size: arguments.model_size,
        threads: arguments.threads,
        detector_postprocess: DetectorPostprocessOptions {
            binary_threshold: arguments.binary_threshold,
            box_threshold: arguments.box_threshold,
            min_area: arguments.min_area,
            unclip_ratio: arguments.unclip_ratio,
            max_boxes: arguments.max_boxes,
        },
        detector_max_side: arguments.detector_max_side,
        recognizer_max_width: arguments.recognizer_max_width,
    };
    options.validate()?;
    let (detector_model, recognizer_model, dictionary) = resolve_models(&arguments)?;
    let engine = OcrEngine::load(detector_model, &recognizer_model, dictionary, options)?;
    let result = engine.recognize_path(&arguments.image)?;
    let output = match arguments.format {
        OutputFormat::Json => {
            serde_json::to_string_pretty(&result).context("serialize OCR result")?
        }
        OutputFormat::Text => result.text(),
    };
    if let Some(path) = &arguments.output {
        fs::write(path, output).with_context(|| format!("write OCR result {}", path.display()))?;
    } else {
        println!("{output}");
    }
    Ok(())
}

fn resolve_models(arguments: &Arguments) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let store = ModelStore::new(&arguments.model_store.model_dir);
    let detector = match &arguments.detector_model {
        Some(path) => path.clone(),
        None => resolve_pinned_model(&store, ModelKind::Detector, arguments)?,
    };
    let recognizer_paths = match &arguments.recognizer_model {
        Some(_) => None,
        None => Some(resolve_pinned_paths(
            &store,
            ModelKind::Recognizer,
            arguments,
        )?),
    };
    let recognizer = match (&arguments.recognizer_model, &recognizer_paths) {
        (Some(path), _) => path.clone(),
        (None, Some(paths)) => paths.weights.clone(),
        (None, None) => anyhow::bail!("recognizer model could not be resolved"),
    };
    let dictionary = match &arguments.dictionary {
        Some(path) => path.clone(),
        None => match recognizer_paths {
            Some(paths) => paths.inference,
            None => recognizer
                .parent()
                .map(|directory| directory.join("inference.yml"))
                .with_context(|| {
                    format!(
                        "derive a dictionary path from recognizer model {}",
                        recognizer.display()
                    )
                })?,
        },
    };
    Ok((detector, recognizer, dictionary))
}

fn resolve_pinned_model(
    store: &ModelStore,
    kind: ModelKind,
    arguments: &Arguments,
) -> Result<PathBuf> {
    Ok(resolve_pinned_paths(store, kind, arguments)?.weights)
}

fn resolve_pinned_paths(
    store: &ModelStore,
    kind: ModelKind,
    arguments: &Arguments,
) -> Result<ModelPaths> {
    store.resolve(kind, arguments.model_size, arguments.model_store.access())
}

fn default_threads() -> usize {
    OcrOptions::default().threads
}
