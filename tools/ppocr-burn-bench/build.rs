use burn_onnx::{LoadStrategy, ModelGen};
use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
};

fn model_path(variable: &str) -> PathBuf {
    let value = env::var_os(variable).unwrap_or_else(|| {
        panic!("{variable} is required; point it at a fixed-shape ONNX model before building")
    });
    let path = PathBuf::from(value);
    if !path.is_file() {
        panic!(
            "{variable} does not name a readable file: {}",
            path.display()
        );
    }
    path.canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", path.display()))
}

fn stage_model(source: &Path, name: &str) -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let staged_dir = out_dir.join("onnx");
    create_dir_all(&staged_dir).expect("create staged ONNX directory");

    // ModelGen derives generated filenames from the input stem. Stage names keep the
    // detector and recognizer distinct even when both source files are inference.onnx.
    let staged = staged_dir.join(format!("{name}.onnx"));
    let _ = fs::remove_file(&staged);
    fs::copy(source, &staged).unwrap_or_else(|error| {
        panic!("copy {} to {}: {error}", source.display(), staged.display())
    });
    staged
}

fn generate(source: &Path, name: &str) {
    let staged = stage_model(source, name);
    ModelGen::new()
        .input(staged.to_str().expect("ONNX path is UTF-8"))
        .out_dir("generated")
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();
}

fn main() {
    for variable in ["PPOCR_BURN_DET_ONNX", "PPOCR_BURN_REC_ONNX"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let detector = model_path("PPOCR_BURN_DET_ONNX");
    let recognizer = model_path("PPOCR_BURN_REC_ONNX");
    println!("cargo:rerun-if-changed={}", detector.display());
    println!("cargo:rerun-if-changed={}", recognizer.display());

    generate(&detector, "detector");
    generate(&recognizer, "recognizer");
}
