//! Write an ONNX copy whose only graph input has a fixed NCHW shape.

use anyhow::{Context, Result, bail};
use onnx_ir::{GraphProto, ModelProto, TensorProto, TensorShapeProto, ValueInfoProto};
use protobuf::Message;
use std::{
    collections::BTreeSet,
    env,
    ffi::OsString,
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Component, Path, PathBuf},
};

struct Arguments {
    input: PathBuf,
    output: PathBuf,
    shape: [i64; 4],
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    let Some(arguments) = parse_arguments(env::args_os().skip(1))? else {
        return Ok(());
    };

    let input_file = File::open(&arguments.input)
        .with_context(|| format!("open input model {}", arguments.input.display()))?;
    let mut input_file = BufReader::new(input_file);
    let mut model = ModelProto::parse_from_reader(&mut input_file)
        .with_context(|| format!("decode ONNX model {}", arguments.input.display()))?;

    let graph = model.graph.as_mut().context("ONNX model has no graph")?;
    if graph.input.len() != 1 {
        bail!(
            "expected exactly one graph input, found {}",
            graph.input.len()
        );
    }
    set_input_shape(&mut graph.input[0], arguments.shape)?;

    if let Some(parent) = arguments
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    copy_external_tensor_data(&model, &arguments.input, &arguments.output)?;

    let output_file = File::create(&arguments.output)
        .with_context(|| format!("create output model {}", arguments.output.display()))?;
    let mut output_file = BufWriter::new(output_file);
    model
        .write_to_writer(&mut output_file)
        .with_context(|| format!("serialize ONNX model {}", arguments.output.display()))?;
    output_file
        .flush()
        .with_context(|| format!("flush output model {}", arguments.output.display()))?;

    println!(
        "wrote {} with input shape {:?}",
        arguments.output.display(),
        arguments.shape
    );
    Ok(())
}

fn parse_arguments(arguments: impl Iterator<Item = OsString>) -> Result<Option<Arguments>> {
    let mut positionals = Vec::new();
    let mut shape = None;
    let mut arguments = arguments.peekable();

    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--help" | "-h") => {
                print_usage();
                return Ok(None);
            }
            Some("--shape") => {
                if shape.is_some() {
                    bail!("--shape may only be specified once");
                }
                shape = Some(parse_shape(&mut arguments)?);
            }
            Some(flag) if flag.starts_with('-') => {
                bail!("unknown option {flag:?}; pass --help for usage");
            }
            _ => positionals.push(PathBuf::from(argument)),
        }
    }

    if positionals.len() != 2 {
        bail!("expected INPUT OUTPUT; pass --help for usage");
    }

    Ok(Some(Arguments {
        input: positionals.remove(0),
        output: positionals.remove(0),
        shape: shape.context("--shape N C H W is required")?,
    }))
}

fn parse_shape(arguments: &mut impl Iterator<Item = OsString>) -> Result<[i64; 4]> {
    let mut shape = [0; 4];
    for (index, dimension) in shape.iter_mut().enumerate() {
        let value = arguments
            .next()
            .with_context(|| format!("--shape requires 4 values; missing value {}", index + 1))?;
        let value = value
            .into_string()
            .map_err(|_| anyhow::anyhow!("--shape values must be valid UTF-8 positive integers"))?;
        *dimension = value
            .parse::<i64>()
            .with_context(|| format!("--shape value {} must be a positive integer", index + 1))?;
        if *dimension <= 0 {
            bail!("--shape value {} must be positive", index + 1);
        }
    }
    Ok(shape)
}

fn set_input_shape(input: &mut ValueInfoProto, shape: [i64; 4]) -> Result<()> {
    let input_name = if input.name.is_empty() {
        "the graph input".to_owned()
    } else {
        format!("graph input {:?}", input.name)
    };
    let type_proto = input
        .type_
        .as_mut()
        .with_context(|| format!("{input_name} has no type"))?;
    if !type_proto.has_tensor_type() {
        bail!("{input_name} is not a tensor input");
    }

    let tensor_type = type_proto.mut_tensor_type();
    if tensor_type.elem_type == 0 {
        bail!("{input_name} has no tensor element type");
    }
    if tensor_type.shape.is_none() {
        tensor_type.shape = Some(TensorShapeProto::new()).into();
    }
    let tensor_shape = tensor_type
        .shape
        .as_mut()
        .expect("tensor shape was initialized above");
    tensor_shape.dim.clear();
    for value in shape {
        tensor_shape.dim.push(Default::default());
        tensor_shape
            .dim
            .last_mut()
            .expect("dimension was pushed above")
            .set_dim_value(value);
    }
    Ok(())
}

macro_rules! collect_external_data_from_attributes {
    ($attributes:expr, $locations:expr) => {{
        for attribute in $attributes {
            if let Some(tensor) = attribute.t.as_ref() {
                collect_external_data_from_tensor(tensor, $locations)?;
            }
            for tensor in &attribute.tensors {
                collect_external_data_from_tensor(tensor, $locations)?;
            }
            if let Some(tensor) = attribute.sparse_tensor.as_ref() {
                collect_external_data_from_sparse_tensor!(tensor, $locations);
            }
            for tensor in &attribute.sparse_tensors {
                collect_external_data_from_sparse_tensor!(tensor, $locations);
            }
            if let Some(graph) = attribute.g.as_ref() {
                collect_external_data_from_graph(graph, $locations)?;
            }
            for graph in &attribute.graphs {
                collect_external_data_from_graph(graph, $locations)?;
            }
        }
    }};
}

macro_rules! collect_external_data_from_sparse_tensor {
    ($tensor:expr, $locations:expr) => {{
        if let Some(values) = $tensor.values.as_ref() {
            collect_external_data_from_tensor(values, $locations)?;
        }
        if let Some(indices) = $tensor.indices.as_ref() {
            collect_external_data_from_tensor(indices, $locations)?;
        }
    }};
}

fn copy_external_tensor_data(model: &ModelProto, input: &Path, output: &Path) -> Result<()> {
    let locations = external_tensor_data_locations(model)?;
    if locations.is_empty() {
        return Ok(());
    }

    let input_dir = input.parent().unwrap_or_else(|| Path::new("."));
    let output_dir = output.parent().unwrap_or_else(|| Path::new("."));
    for location in locations {
        let source = input_dir.join(&location);
        let destination = output_dir.join(&location);
        if source == destination {
            continue;
        }
        if !source.is_file() {
            bail!(
                "external tensor data sidecar is missing: {}",
                source.display()
            );
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("create external tensor data directory {}", parent.display())
            })?;
        }
        fs::copy(&source, &destination).with_context(|| {
            format!(
                "copy external tensor data {} to {}",
                source.display(),
                destination.display()
            )
        })?;
    }
    Ok(())
}

fn external_tensor_data_locations(model: &ModelProto) -> Result<BTreeSet<PathBuf>> {
    let mut locations = BTreeSet::new();
    if let Some(graph) = model.graph.as_ref() {
        collect_external_data_from_graph(graph, &mut locations)?;
    }
    for training_info in &model.training_info {
        if let Some(graph) = training_info.initialization.as_ref() {
            collect_external_data_from_graph(graph, &mut locations)?;
        }
        if let Some(graph) = training_info.algorithm.as_ref() {
            collect_external_data_from_graph(graph, &mut locations)?;
        }
    }
    for function in &model.functions {
        collect_external_data_from_attributes!(&function.attribute_proto, &mut locations);
        for node in &function.node {
            collect_external_data_from_attributes!(&node.attribute, &mut locations);
        }
    }
    Ok(locations)
}

fn collect_external_data_from_graph(
    graph: &GraphProto,
    locations: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    for tensor in &graph.initializer {
        collect_external_data_from_tensor(tensor, locations)?;
    }
    for tensor in &graph.sparse_initializer {
        collect_external_data_from_sparse_tensor!(tensor, locations);
    }
    for node in &graph.node {
        collect_external_data_from_attributes!(&node.attribute, locations);
    }
    Ok(())
}

fn collect_external_data_from_tensor(
    tensor: &TensorProto,
    locations: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    if tensor.external_data.is_empty() && tensor.data_location.value() == 0 {
        return Ok(());
    }
    let location = tensor
        .external_data
        .iter()
        .find(|entry| entry.key == "location")
        .map(|entry| entry.value.as_str())
        .context("external tensor data has no location entry")?;
    locations.insert(relative_sidecar_path(location)?);
    Ok(())
}

fn relative_sidecar_path(location: &str) -> Result<PathBuf> {
    let path = Path::new(location);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!(
            "external tensor data location must be a relative path without parent traversal: {location:?}"
        );
    }
    let path = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(component) => Some(component),
            Component::CurDir => None,
            _ => None,
        })
        .collect::<PathBuf>();
    if path.as_os_str().is_empty() {
        bail!("external tensor data location must not be empty");
    }
    Ok(path)
}

fn print_usage() {
    println!("usage: ppocr-onnx-staticize INPUT OUTPUT --shape N C H W");
}

#[cfg(test)]
mod tests {
    use super::{relative_sidecar_path, set_input_shape};
    use onnx_ir::{TensorShapeProto, TypeProto, ValueInfoProto};

    #[test]
    fn accepts_a_nested_relative_sidecar_path() {
        assert_eq!(
            relative_sidecar_path("weights/model.data").unwrap(),
            std::path::Path::new("weights/model.data")
        );
    }

    #[test]
    fn rejects_parent_traversal_in_a_sidecar_path() {
        assert!(relative_sidecar_path("../model.data").is_err());
    }

    #[test]
    fn replaces_dynamic_nchw_dimensions() {
        let mut input = ValueInfoProto::new();
        input.name = "images".to_owned();
        let mut type_proto = TypeProto::new();
        let tensor_type = type_proto.mut_tensor_type();
        tensor_type.elem_type = 1;
        let mut tensor_shape = TensorShapeProto::new();
        tensor_shape.dim.resize_with(4, Default::default);
        tensor_type.shape = Some(tensor_shape).into();
        input.type_ = Some(type_proto).into();

        set_input_shape(&mut input, [1, 3, 416, 736]).unwrap();

        let dimensions = input
            .type_
            .as_ref()
            .unwrap()
            .tensor_type()
            .shape
            .as_ref()
            .unwrap()
            .dim
            .iter()
            .map(|dimension| dimension.dim_value())
            .collect::<Vec<_>>();
        assert_eq!(dimensions, [1, 3, 416, 736]);
    }
}
