#[cfg(feature = "burn-infer")]
use burn_onnx::{LoadStrategy, ModelGen};
#[cfg(feature = "burn-infer")]
use onnx_ir_build::ModelProto;
#[cfg(feature = "burn-infer")]
use protobuf_build::Message;
#[cfg(feature = "burn-infer")]
use std::{
    env,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

#[cfg(feature = "burn-infer")]
fn model_path(variable: &str) -> Option<PathBuf> {
    let value = env::var_os(variable)?;
    let path = PathBuf::from(value);
    if !path.is_file() {
        panic!(
            "{variable} does not name a readable file: {}",
            path.display()
        );
    }
    Some(
        path.canonicalize()
            .unwrap_or_else(|error| panic!("canonicalize {}: {error}", path.display())),
    )
}

#[cfg(feature = "burn-infer")]
fn generate(source: &Path, name: &str) {
    let stem = source
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_else(|| panic!("ONNX path is not a UTF-8 filename: {}", source.display()));
    let out_dir = format!("generated/{name}");
    ModelGen::new()
        .input(source.to_str().expect("ONNX path is UTF-8"))
        .out_dir(&out_dir)
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();

    let generated = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join(out_dir)
        .join(format!("{stem}.rs"));
    println!(
        "cargo:rustc-env=PPOCR_BURN_{}_MODEL_RS={}",
        name.to_ascii_uppercase(),
        generated.display()
    );
}

#[cfg(feature = "burn-infer")]
fn recognizer_width(source: &Path) -> usize {
    let input = File::open(source)
        .unwrap_or_else(|error| panic!("open recognizer ONNX {}: {error}", source.display()));
    let mut input = BufReader::new(input);
    let model = ModelProto::parse_from_reader(&mut input)
        .unwrap_or_else(|error| panic!("decode recognizer ONNX {}: {error}", source.display()));
    let graph = model
        .graph
        .as_ref()
        .unwrap_or_else(|| panic!("recognizer ONNX {} has no graph", source.display()));
    if graph.input.len() != 1 {
        panic!(
            "recognizer ONNX {} must have exactly one graph input, found {}",
            source.display(),
            graph.input.len()
        );
    }
    let input = &graph.input[0];
    let type_proto = input.type_.as_ref().unwrap_or_else(|| {
        panic!(
            "recognizer ONNX {} input {:?} has no type",
            source.display(),
            input.name
        )
    });
    if !type_proto.has_tensor_type() {
        panic!(
            "recognizer ONNX {} input {:?} is not a tensor",
            source.display(),
            input.name
        );
    }
    let tensor_type = type_proto.tensor_type();
    let shape = tensor_type.shape.as_ref().unwrap_or_else(|| {
        panic!(
            "recognizer ONNX {} input {:?} has no shape",
            source.display(),
            input.name
        )
    });
    let dimensions = shape
        .dim
        .iter()
        .map(|dimension| dimension.dim_value())
        .collect::<Vec<_>>();
    if dimensions.len() != 4
        || dimensions[0] != 1
        || dimensions[1] != 3
        || dimensions[2] != 48
        || dimensions[3] <= 0
    {
        panic!(
            "recognizer ONNX {} input {:?} must have fixed shape [1, 3, 48, width], found {dimensions:?}",
            source.display(),
            input.name
        );
    }
    usize::try_from(dimensions[3]).unwrap_or_else(|_| {
        panic!(
            "recognizer ONNX {} width {} does not fit usize",
            source.display(),
            dimensions[3]
        )
    })
}

#[cfg(feature = "burn-infer")]
fn main() {
    println!("cargo:rustc-check-cfg=cfg(ppocr_burn_models)");
    for variable in ["PPOCR_BURN_DET_ONNX", "PPOCR_BURN_REC_ONNX"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    let (detector, recognizer) = match (
        model_path("PPOCR_BURN_DET_ONNX"),
        model_path("PPOCR_BURN_REC_ONNX"),
    ) {
        (Some(detector), Some(recognizer)) => (detector, recognizer),
        (None, None) => {
            println!(
                "cargo:warning=Burn model generation skipped; set PPOCR_BURN_DET_ONNX and PPOCR_BURN_REC_ONNX to build Burn inference or benchmarks"
            );
            return;
        }
        _ => panic!("PPOCR_BURN_DET_ONNX and PPOCR_BURN_REC_ONNX must be set together"),
    };
    println!("cargo:rerun-if-changed={}", detector.display());
    println!("cargo:rerun-if-changed={}", recognizer.display());

    let recognizer_width = recognizer_width(&recognizer);
    println!("cargo:rustc-env=PPOCR_BURN_RECOGNIZER_WIDTH={recognizer_width}");

    generate(&detector, "detector");
    generate(&recognizer, "recognizer");
    println!("cargo:rustc-cfg=ppocr_burn_models");
}

#[cfg(not(feature = "burn-infer"))]
fn main() {}
