use anyhow::{Context, Result, bail};
use rten_onnx::onnx::{AttributeProto, ModelProto};
use std::{
    collections::{BTreeMap, HashMap},
    env,
    fs::File,
    path::Path,
};

fn main() -> Result<()> {
    let mut arguments = env::args().skip(1);
    let path = arguments
        .next()
        .context("usage: ppocr-cpu-inspect MODEL.onnx")?;
    let conv_shapes = arguments.next().as_deref() == Some("--conv-shapes");
    inspect(Path::new(&path), conv_shapes)
}

fn inspect(path: &Path, conv_shapes: bool) -> Result<()> {
    let model = ModelProto::parse_file(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    )
    .with_context(|| format!("decode {}", path.display()))?;
    let graph = model.graph.context("ONNX model has no graph")?;
    if graph.input.len() != 1 || graph.output.len() != 1 {
        bail!(
            "expected one graph input and output, found {} and {}",
            graph.input.len(),
            graph.output.len()
        );
    }

    let mut counts = BTreeMap::<&str, usize>::new();
    for node in &graph.node {
        *counts
            .entry(node.op_type.as_deref().unwrap_or("<missing>"))
            .or_default() += 1;
    }
    println!(
        "{}: {} nodes, {} initializers",
        path.display(),
        graph.node.len(),
        graph.initializer.len()
    );
    for (op, count) in counts {
        println!("{op:24} {count:4}");
    }
    if conv_shapes {
        let initializers = graph
            .initializer
            .iter()
            .filter_map(|tensor| {
                tensor
                    .name
                    .as_deref()
                    .map(|name| (name, tensor.dims.as_slice()))
            })
            .collect::<HashMap<_, _>>();
        for node in &graph.node {
            if matches!(node.op_type.as_deref(), Some("Conv" | "ConvTranspose")) {
                let shape = node
                    .input
                    .get(1)
                    .and_then(|name| initializers.get(name.as_str()))
                    .copied();
                println!(
                    "{} {} weight={shape:?}",
                    node.op_type.as_deref().unwrap_or("<missing>"),
                    node.name.as_deref().unwrap_or("<unnamed>"),
                );
            }
        }
        return Ok(());
    }
    println!("attributes:");
    for node in &graph.node {
        if !node.attribute.is_empty() {
            let attributes = node
                .attribute
                .iter()
                .map(format_attribute)
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "{} {}: {}",
                node.op_type.as_deref().unwrap_or("<missing>"),
                node.name.as_deref().unwrap_or("<unnamed>"),
                attributes
            );
        }
    }
    Ok(())
}

fn format_attribute(attribute: &AttributeProto) -> String {
    let name = attribute.name.as_deref().unwrap_or("<missing>");
    if let Some(value) = attribute.i {
        format!("{name}={value}")
    } else if let Some(value) = attribute.f {
        format!("{name}={value}")
    } else if let Some(value) = &attribute.s {
        format!("{name}={value:?}")
    } else if !attribute.ints.is_empty() {
        format!("{name}={:?}", attribute.ints)
    } else if !attribute.floats.is_empty() {
        format!("{name}={:?}", attribute.floats)
    } else {
        format!("{name}=<tensor-or-graph>")
    }
}
