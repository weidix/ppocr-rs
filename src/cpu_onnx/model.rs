use crate::cpu_onnx::{
    ops::{ConvOptions, Node, Operation, PoolOptions, ValueId},
    tensor::{Tensor, element_count},
};

#[cfg(feature = "cpu_onnx-convert")]
use crate::cpu_onnx::tensor::TensorData;
use anyhow::{Context, Result, bail, ensure};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

#[cfg(feature = "cpu_onnx-convert")]
use std::io::{BufWriter, Write};

const MAGIC: &[u8; 8] = b"PPOCRCPU";
const FORMAT_VERSION: u32 = 1;
const NONE_VALUE_ID: u32 = u32::MAX;
const MAX_RANK: usize = 16;
const MAX_NODES: usize = 10_000;
const MAX_VALUES: usize = 20_000;

#[derive(Clone, Copy, Debug)]
pub struct CpuOptions {
    pub threads: usize,
}

impl Default for CpuOptions {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(4),
        }
    }
}

pub struct CpuModel {
    data: ModelData,
    pool: ThreadPool,
}

struct ModelData {
    input: ValueId,
    output: ValueId,
    input_shape: Vec<usize>,
    initial_values: Vec<Option<Tensor>>,
    nodes: Vec<Node>,
    use_counts: Vec<usize>,
}

impl CpuModel {
    pub fn load(path: impl AsRef<Path>, options: CpuOptions) -> Result<Self> {
        ensure!(options.threads > 0, "CPU thread count must be positive");
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("open model {}", path.display()))?;
        let mut data = ModelData::read(&mut BufReader::new(file))
            .with_context(|| format!("decode model {}", path.display()))?;
        data.prepare_pointwise_weights()?;
        let pool = ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .thread_name(|index| format!("ppocr-cpu-onnx-{index}"))
            .build()
            .context("create CPU inference thread pool")?;
        Ok(Self { data, pool })
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.data.input_shape
    }

    pub fn run(&self, input: Tensor) -> Result<Tensor> {
        ensure!(
            input.shape() == self.data.input_shape,
            "model expects input shape {:?}, found {:?}",
            self.data.input_shape,
            input.shape()
        );
        self.pool.install(|| self.data.run(input))
    }
}

impl ModelData {
    fn prepare_pointwise_weights(&mut self) -> Result<()> {
        use std::collections::HashSet;

        let mut packed = HashSet::new();
        for node in &mut self.nodes {
            let (Operation::Conv(options) | Operation::ConvGelu(options)) = &mut node.operation
            else {
                continue;
            };
            if options.groups != 1 {
                continue;
            }
            let weight_id = node
                .inputs
                .get(1)
                .copied()
                .flatten()
                .context("Conv node has no weight input")?;
            let weight = self.initial_values[weight_id]
                .as_ref()
                .context("Conv weight is not an initializer")?;
            if weight.shape.len() != 4
                || (weight.shape[2..] != [1, 1]
                    && !(options.strides != [1, 1] && weight.shape[0] >= 48))
            {
                continue;
            }
            options.packed_pointwise = true;
            if !packed.insert(weight_id) {
                continue;
            }
            let rows = weight.shape[0];
            let source = weight.as_f32()?;
            let inner = source
                .len()
                .checked_div(rows)
                .context("Conv weight has zero output channels")?;
            let mut values = Vec::with_capacity(source.len());
            for row_start in (0..rows).step_by(12) {
                let block_rows = (rows - row_start).min(12);
                for index in 0..inner {
                    for row in 0..block_rows {
                        values.push(source[(row_start + row) * inner + index]);
                    }
                }
            }
            self.initial_values[weight_id] = Some(Tensor::new_f32(weight.shape.clone(), values));
        }
        Ok(())
    }

    fn run(&self, input: Tensor) -> Result<Tensor> {
        let mut values = self.initial_values.clone();
        values[self.input] = Some(input);
        let mut uses = self.use_counts.clone();
        let profile = std::env::var_os("PPOCR_CPU_ONNX_PROFILE").is_some();
        for node in &self.nodes {
            let start = profile.then(std::time::Instant::now);
            let mut inputs = Vec::with_capacity(node.inputs.len());
            for &input_id in node.inputs.iter().flatten() {
                let value = values[input_id].as_ref().with_context(|| {
                    format!("{} references unavailable value {input_id}", node.name)
                })?;
                if uses[input_id] == 1 {
                    inputs.push(values[input_id].take().expect("value checked above"));
                } else {
                    inputs.push(value.clone());
                }
                uses[input_id] = uses[input_id]
                    .checked_sub(1)
                    .with_context(|| format!("invalid use count for value {input_id}"))?;
            }
            let output = node.run(inputs)?;
            let output_shape = profile.then(|| output.shape().to_vec());
            ensure!(
                values[node.output].is_none(),
                "{} writes value {} twice",
                node.name,
                node.output
            );
            values[node.output] = Some(output);
            if let Some(start) = start {
                eprintln!(
                    "{:.6}\t{}\t{:?}",
                    start.elapsed().as_secs_f64() * 1_000.0,
                    node.name,
                    output_shape.expect("profile shape captured")
                );
            }
        }
        values[self.output]
            .take()
            .context("model did not produce its graph output")
    }

    fn finish(mut self) -> Result<Self> {
        ensure!(
            self.initial_values.len() <= MAX_VALUES,
            "model has too many values"
        );
        ensure!(self.nodes.len() <= MAX_NODES, "model has too many nodes");
        let mut use_counts = vec![0usize; self.initial_values.len()];
        for node in &self.nodes {
            ensure!(
                node.output < use_counts.len(),
                "node output is out of range"
            );
            for &input in node.inputs.iter().flatten() {
                ensure!(input < use_counts.len(), "node input is out of range");
                use_counts[input] += 1;
            }
        }
        use_counts[self.output] += 1;
        self.use_counts = use_counts;
        Ok(self)
    }

    #[cfg(feature = "cpu_onnx-convert")]
    fn optimize(mut self) -> Result<Self> {
        self.fuse_conv_biases()?;
        self.fold_conv_batch_normalization()?;
        let uses = value_uses(&self.nodes, self.initial_values.len());
        let mut optimized = Vec::with_capacity(self.nodes.len());
        let mut index = 0;
        while index < self.nodes.len() {
            if let Some(node) = match_gelu(&self.nodes[index..], &self.initial_values, &uses) {
                optimized.push(node);
                index += 5;
                continue;
            }
            if let Some((node, consumed)) = match_gated_activation(&self.nodes[index..], &uses) {
                optimized.push(node);
                index += consumed;
                continue;
            }
            optimized.push(self.nodes[index].clone());
            index += 1;
        }
        let uses = value_uses(&optimized, self.initial_values.len());
        let mut fused = Vec::with_capacity(optimized.len());
        let mut index = 0;
        while index < optimized.len() {
            let conv = &optimized[index];
            let gelu = optimized.get(index + 1);
            if let (Operation::Conv(options), Some(gelu)) = (&conv.operation, gelu)
                && matches!(gelu.operation, Operation::Gelu)
                && uses[conv.output] == 1
                && node_inputs(gelu).as_deref() == Some(&[conv.output])
            {
                fused.push(Node {
                    name: format!("FusedConvGelu.{}", conv.name),
                    inputs: conv.inputs.clone(),
                    output: gelu.output,
                    operation: Operation::ConvGelu(options.clone()),
                });
                index += 2;
            } else {
                fused.push(conv.clone());
                index += 1;
            }
        }
        let uses = value_uses(&fused, self.initial_values.len());
        let mut optimized = Vec::with_capacity(fused.len());
        let mut index = 0;
        while index < fused.len() {
            let add = &fused[index];
            let softmax = fused.get(index + 1);
            let inputs = node_inputs(add);
            let bias_and_data = inputs.as_deref().and_then(|inputs| {
                if inputs.len() != 2 {
                    return None;
                }
                let first_constant = self.initial_values[inputs[0]].is_some();
                let second_constant = self.initial_values[inputs[1]].is_some();
                match (first_constant, second_constant) {
                    (true, false) => Some((inputs[0], inputs[1])),
                    (false, true) => Some((inputs[1], inputs[0])),
                    _ => None,
                }
            });
            if let (Operation::Add, Some(softmax), Some((bias, data))) =
                (&add.operation, softmax, bias_and_data)
                && let Operation::Softmax { axis } = softmax.operation
                && uses[add.output] == 1
                && node_inputs(softmax).is_some_and(|inputs| inputs == [add.output])
            {
                optimized.push(Node {
                    name: format!("FusedBiasSoftmax.{}", add.name),
                    inputs: vec![Some(data), Some(bias)],
                    output: softmax.output,
                    operation: Operation::BiasSoftmax { axis },
                });
                index += 2;
            } else {
                optimized.push(add.clone());
                index += 1;
            }
        }
        let uses = value_uses(&optimized, self.initial_values.len());
        let mut fused = Vec::with_capacity(optimized.len());
        let mut index = 0;
        while index < optimized.len() {
            let matmul = &optimized[index];
            let bias_softmax = optimized.get(index + 1);
            if let (Operation::MatMul, Some(bias_softmax)) = (&matmul.operation, bias_softmax)
                && let Operation::BiasSoftmax { axis } = bias_softmax.operation
                && uses[matmul.output] == 1
                && node_inputs(bias_softmax)
                    .is_some_and(|inputs| inputs.len() == 2 && inputs[0] == matmul.output)
            {
                let mut inputs = matmul.inputs.clone();
                inputs.push(bias_softmax.inputs[1]);
                fused.push(Node {
                    name: format!("FusedMatMulBiasSoftmax.{}", matmul.name),
                    inputs,
                    output: bias_softmax.output,
                    operation: Operation::MatMulBiasSoftmax { axis },
                });
                index += 2;
            } else {
                fused.push(matmul.clone());
                index += 1;
            }
        }
        self.nodes = fused;
        Ok(self)
    }

    #[cfg(feature = "cpu_onnx-convert")]
    fn fuse_conv_biases(&mut self) -> Result<()> {
        let uses = value_uses(&self.nodes, self.initial_values.len());
        let mut fused = Vec::with_capacity(self.nodes.len());
        let mut index = 0;
        while index < self.nodes.len() {
            let Some(add) = self.nodes.get(index + 1) else {
                fused.push(self.nodes[index].clone());
                break;
            };
            let mut conv = self.nodes[index].clone();
            let add_inputs = node_inputs(add);
            let bias_id = add_inputs
                .as_deref()
                .filter(|_| matches!(conv.operation, Operation::Conv(_)))
                .filter(|_| matches!(add.operation, Operation::Add))
                .filter(|_| uses[conv.output] == 1)
                .and_then(|inputs| other_input(inputs, conv.output))
                .filter(|&id| self.initial_values[id].is_some());
            let Some(bias_id) = bias_id else {
                fused.push(conv);
                index += 1;
                continue;
            };
            if conv.inputs.get(2).is_some_and(Option::is_some) {
                fused.push(conv);
                index += 1;
                continue;
            }
            let weight_id = conv
                .inputs
                .get(1)
                .copied()
                .flatten()
                .context("Conv node has no weight input")?;
            let output_channels = self.initial_values[weight_id]
                .as_ref()
                .context("Conv weight is not an initializer")?
                .shape
                .first()
                .copied()
                .context("Conv weight has no output-channel dimension")?;
            if self.initial_values[bias_id]
                .as_ref()
                .context("Conv bias is unavailable")?
                .as_f32()?
                .len()
                != output_channels
            {
                fused.push(conv);
                index += 1;
                continue;
            }
            if conv.inputs.len() == 2 {
                conv.inputs.push(Some(bias_id));
            } else {
                conv.inputs[2] = Some(bias_id);
            }
            conv.output = add.output;
            conv.name = format!("FusedConvBias.{}", conv.name);
            fused.push(conv);
            index += 2;
        }
        self.nodes = fused;
        Ok(())
    }

    #[cfg(feature = "cpu_onnx-convert")]
    fn fold_conv_batch_normalization(&mut self) -> Result<()> {
        let uses = value_uses(&self.nodes, self.initial_values.len());
        let mut fused = Vec::with_capacity(self.nodes.len());
        let mut index = 0;
        while index < self.nodes.len() {
            let Some(next) = self.nodes.get(index + 1) else {
                fused.push(self.nodes[index].clone());
                break;
            };
            let mut conv = self.nodes[index].clone();
            let (batch_norm, squeeze) =
                if matches!(next.operation, Operation::BatchNormalization { .. }) {
                    (next, None)
                } else if matches!(&next.operation, Operation::Squeeze { axes } if axes == &[2])
                    && uses[conv.output] == 1
                    && node_inputs(next).is_some_and(|inputs| inputs.first() == Some(&conv.output))
                {
                    let Some(batch_norm) = self.nodes.get(index + 2) else {
                        fused.push(conv);
                        index += 1;
                        continue;
                    };
                    (batch_norm, Some(next))
                } else {
                    fused.push(conv);
                    index += 1;
                    continue;
                };
            let Operation::BatchNormalization { epsilon } = batch_norm.operation else {
                fused.push(conv);
                index += 1;
                continue;
            };
            let Some(inputs) = node_inputs(batch_norm) else {
                fused.push(conv);
                index += 1;
                continue;
            };
            let batch_norm_input = squeeze.map_or(conv.output, |squeeze| squeeze.output);
            if !matches!(conv.operation, Operation::Conv(_))
                || uses[conv.output] != 1
                || squeeze.is_some_and(|squeeze| uses[squeeze.output] != 1)
                || inputs.len() != 5
                || inputs[0] != batch_norm_input
            {
                fused.push(conv);
                index += 1;
                continue;
            }

            let weight_id = conv
                .inputs
                .get(1)
                .copied()
                .flatten()
                .context("Conv node has no weight input")?;
            let weight = self.initial_values[weight_id]
                .as_ref()
                .context("Conv weight is not an initializer")?;
            let output_channels = weight
                .shape
                .first()
                .copied()
                .context("Conv weight has no output-channel dimension")?;
            let channel_size = weight
                .len()
                .checked_div(output_channels)
                .context("Conv has zero output channels")?;
            ensure!(
                channel_size * output_channels == weight.len(),
                "Conv weight size is not divisible by output channels"
            );
            let scale = initializer_f32(&self.initial_values, inputs[1], output_channels)?;
            let offset = initializer_f32(&self.initial_values, inputs[2], output_channels)?;
            let mean = initializer_f32(&self.initial_values, inputs[3], output_channels)?;
            let variance = initializer_f32(&self.initial_values, inputs[4], output_channels)?;
            let old_bias = conv
                .inputs
                .get(2)
                .copied()
                .flatten()
                .map(|id| initializer_f32(&self.initial_values, id, output_channels))
                .transpose()?;

            let mut multipliers = Vec::with_capacity(output_channels);
            let mut bias = Vec::with_capacity(output_channels);
            for channel in 0..output_channels {
                let multiplier = scale[channel] / (variance[channel] + epsilon).sqrt();
                multipliers.push(multiplier);
                bias.push(
                    (old_bias.as_ref().map_or(0.0, |bias| bias[channel]) - mean[channel])
                        .mul_add(multiplier, offset[channel]),
                );
            }
            let mut weight_values = weight.as_f32()?.to_vec();
            for (channel, values) in weight_values.chunks_mut(channel_size).enumerate() {
                for value in values {
                    *value *= multipliers[channel];
                }
            }
            let new_weight_id = self.initial_values.len();
            self.initial_values
                .push(Some(Tensor::new_f32(weight.shape.clone(), weight_values)));
            let new_bias_id = self.initial_values.len();
            self.initial_values
                .push(Some(Tensor::new_f32(vec![output_channels], bias)));
            conv.inputs[1] = Some(new_weight_id);
            if conv.inputs.len() == 2 {
                conv.inputs.push(Some(new_bias_id));
            } else {
                conv.inputs[2] = Some(new_bias_id);
            }
            conv.name = format!("FusedConvBatchNorm.{}", conv.name);
            if let Some(squeeze) = squeeze {
                let mut squeeze = squeeze.clone();
                squeeze.output = batch_norm.output;
                fused.push(conv);
                fused.push(squeeze);
                index += 3;
            } else {
                conv.output = batch_norm.output;
                fused.push(conv);
                index += 2;
            }
        }
        self.nodes = fused;
        Ok(())
    }

    #[cfg(feature = "cpu_onnx-convert")]
    fn write(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(MAGIC)?;
        write_u32(writer, FORMAT_VERSION)?;
        write_u32(writer, to_u32(self.initial_values.len(), "value count")?)?;
        write_u32(writer, to_u32(self.input, "input id")?)?;
        write_u32(writer, to_u32(self.output, "output id")?)?;
        write_usize_vec(writer, &self.input_shape)?;

        let initial_count = self.initial_values.iter().flatten().count();
        write_u32(writer, to_u32(initial_count, "initializer count")?)?;
        for (id, tensor) in self
            .initial_values
            .iter()
            .enumerate()
            .filter_map(|(id, tensor)| tensor.as_ref().map(|tensor| (id, tensor)))
        {
            write_u32(writer, to_u32(id, "initializer id")?)?;
            write_usize_vec(writer, &tensor.shape)?;
            match &tensor.data {
                TensorData::F32(values) => {
                    writer.write_all(&[1])?;
                    write_u64(writer, values.len() as u64)?;
                    for value in values.iter() {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
                TensorData::I64(values) => {
                    writer.write_all(&[2])?;
                    write_u64(writer, values.len() as u64)?;
                    for value in values.iter() {
                        writer.write_all(&value.to_le_bytes())?;
                    }
                }
            }
        }

        write_u32(writer, to_u32(self.nodes.len(), "node count")?)?;
        for node in &self.nodes {
            write_string(writer, &node.name)?;
            write_u32(writer, to_u32(node.output, "node output")?)?;
            write_u32(writer, to_u32(node.inputs.len(), "node input count")?)?;
            for input in &node.inputs {
                write_u32(
                    writer,
                    input
                        .map(|id| to_u32(id, "node input"))
                        .transpose()?
                        .unwrap_or(NONE_VALUE_ID),
                )?;
            }
            write_operation(writer, &node.operation)?;
        }
        Ok(())
    }

    fn read(reader: &mut impl Read) -> Result<Self> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "not a ppocr CPU model");
        ensure!(
            read_u32(reader)? == FORMAT_VERSION,
            "unsupported ppocr CPU model version"
        );
        let value_count = read_len(reader, MAX_VALUES, "value count")?;
        let input = read_id(reader, value_count, "input id")?;
        let output = read_id(reader, value_count, "output id")?;
        let input_shape = read_usize_vec(reader, MAX_RANK, "input shape")?;
        let mut initial_values = vec![None; value_count];
        let initializer_count = read_len(reader, value_count, "initializer count")?;
        for _ in 0..initializer_count {
            let id = read_id(reader, value_count, "initializer id")?;
            ensure!(
                initial_values[id].is_none(),
                "duplicate initializer id {id}"
            );
            let shape = read_usize_vec(reader, MAX_RANK, "initializer shape")?;
            let expected = element_count(&shape).context("initializer shape overflow")?;
            let mut data_type = [0u8; 1];
            reader.read_exact(&mut data_type)?;
            let length = usize::try_from(read_u64(reader)?)?;
            ensure!(
                length == expected,
                "initializer data length does not match shape"
            );
            let tensor = match data_type[0] {
                1 => {
                    let mut values = Vec::with_capacity(length);
                    for _ in 0..length {
                        let mut bytes = [0u8; 4];
                        reader.read_exact(&mut bytes)?;
                        values.push(f32::from_le_bytes(bytes));
                    }
                    Tensor::new_f32(shape, values)
                }
                2 => {
                    let mut values = Vec::with_capacity(length);
                    for _ in 0..length {
                        let mut bytes = [0u8; 8];
                        reader.read_exact(&mut bytes)?;
                        values.push(i64::from_le_bytes(bytes));
                    }
                    Tensor::new_i64(shape, values)
                }
                value => bail!("unsupported initializer data type {value}"),
            };
            initial_values[id] = Some(tensor);
        }
        let node_count = read_len(reader, MAX_NODES, "node count")?;
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let name = read_string(reader)?;
            let node_output = read_id(reader, value_count, "node output")?;
            let input_count = read_len(reader, 16, "node input count")?;
            let mut inputs = Vec::with_capacity(input_count);
            for _ in 0..input_count {
                let id = read_u32(reader)?;
                inputs.push(if id == NONE_VALUE_ID {
                    None
                } else {
                    let id = usize::try_from(id)?;
                    ensure!(id < value_count, "node input id is out of range");
                    Some(id)
                });
            }
            nodes.push(Node {
                name,
                inputs,
                output: node_output,
                operation: read_operation(reader)?,
            });
        }
        Self {
            input,
            output,
            input_shape,
            initial_values,
            nodes,
            use_counts: Vec::new(),
        }
        .finish()
    }
}

#[cfg(feature = "cpu_onnx-convert")]
fn value_uses(nodes: &[Node], value_count: usize) -> Vec<usize> {
    let mut uses = vec![0usize; value_count];
    for node in nodes {
        for &input in node.inputs.iter().flatten() {
            uses[input] += 1;
        }
    }
    uses
}

#[cfg(feature = "cpu_onnx-convert")]
fn initializer_f32(values: &[Option<Tensor>], id: ValueId, expected_len: usize) -> Result<&[f32]> {
    let value = values
        .get(id)
        .and_then(Option::as_ref)
        .with_context(|| format!("value {id} is not an initializer"))?
        .as_f32()?;
    ensure!(
        value.len() == expected_len,
        "initializer {id} has length {}, expected {expected_len}",
        value.len()
    );
    Ok(value)
}

#[cfg(feature = "cpu_onnx-convert")]
fn match_gelu(nodes: &[Node], values: &[Option<Tensor>], uses: &[usize]) -> Option<Node> {
    let [div, erf, add, mul, scale, ..] = nodes else {
        return None;
    };
    if !matches!(div.operation, Operation::Div)
        || !matches!(erf.operation, Operation::Erf)
        || !matches!(add.operation, Operation::Add)
        || !matches!(mul.operation, Operation::Mul)
        || !matches!(scale.operation, Operation::Mul)
        || [div.output, erf.output, add.output, mul.output]
            .into_iter()
            .any(|output| uses[output] != 1)
    {
        return None;
    }
    let div_inputs = node_inputs(div)?;
    if div_inputs.len() != 2
        || scalar(values, div_inputs[1])
            .is_none_or(|value| (value - std::f32::consts::SQRT_2).abs() > 1e-5)
        || node_inputs(erf)? != [div.output]
    {
        return None;
    }
    let add_inputs = node_inputs(add)?;
    let add_scalar = other_input(&add_inputs, erf.output)?;
    if scalar(values, add_scalar).is_none_or(|value| (value - 1.0).abs() > 1e-6) {
        return None;
    }
    if !has_inputs(mul, div_inputs[0], add.output) {
        return None;
    }
    let scale_inputs = node_inputs(scale)?;
    let scale_scalar = other_input(&scale_inputs, mul.output)?;
    if scalar(values, scale_scalar).is_none_or(|value| (value - 0.5).abs() > 1e-6) {
        return None;
    }
    Some(Node {
        name: format!("FusedGelu.{}", div.name),
        inputs: vec![Some(div_inputs[0])],
        output: scale.output,
        operation: Operation::Gelu,
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn match_gated_activation(nodes: &[Node], uses: &[usize]) -> Option<(Node, usize)> {
    let [gate, mul, ..] = nodes else {
        return None;
    };
    if uses[gate.output] != 1 || !matches!(mul.operation, Operation::Mul) {
        return None;
    }
    let input = *node_inputs(gate)?.first()?;
    if !has_inputs(mul, input, gate.output) {
        return None;
    }
    let operation = match gate.operation {
        Operation::Sigmoid => Operation::Silu,
        Operation::HardSigmoid { alpha, beta }
            if (alpha - 1.0 / 6.0).abs() < 1e-5 && (beta - 0.5).abs() < 1e-6 =>
        {
            Operation::HardSwish
        }
        _ => return None,
    };
    Some((
        Node {
            name: format!("FusedActivation.{}", gate.name),
            inputs: vec![Some(input)],
            output: mul.output,
            operation,
        },
        2,
    ))
}

#[cfg(feature = "cpu_onnx-convert")]
fn node_inputs(node: &Node) -> Option<Vec<ValueId>> {
    node.inputs.iter().copied().collect()
}

#[cfg(feature = "cpu_onnx-convert")]
fn has_inputs(node: &Node, left: ValueId, right: ValueId) -> bool {
    node_inputs(node).is_some_and(|inputs| {
        inputs.len() == 2
            && ((inputs[0] == left && inputs[1] == right)
                || (inputs[0] == right && inputs[1] == left))
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn other_input(inputs: &[ValueId], known: ValueId) -> Option<ValueId> {
    (inputs.len() == 2).then_some(())?;
    if inputs[0] == known {
        Some(inputs[1])
    } else if inputs[1] == known {
        Some(inputs[0])
    } else {
        None
    }
}

#[cfg(feature = "cpu_onnx-convert")]
fn scalar(values: &[Option<Tensor>], id: ValueId) -> Option<f32> {
    let tensor = values.get(id)?.as_ref()?;
    let values = tensor.as_f32().ok()?;
    (values.len() == 1).then_some(values[0])
}

#[cfg(feature = "cpu_onnx-convert")]
pub fn convert_onnx(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    input_shape: &[usize],
) -> Result<()> {
    use rten_onnx::onnx::ModelProto;

    ensure!(
        input_shape.len() == 4,
        "PP-OCR input shape must have rank four"
    );
    ensure!(
        input_shape.iter().all(|&dimension| dimension > 0),
        "input dimensions must be positive"
    );
    let input = input.as_ref();
    let model = ModelProto::parse_file(
        File::open(input).with_context(|| format!("open ONNX model {}", input.display()))?,
    )
    .with_context(|| format!("decode ONNX model {}", input.display()))?;
    let data = compile_onnx(model, input_shape)?.optimize()?.finish()?;
    let output = output.as_ref();
    let file =
        File::create(output).with_context(|| format!("create model {}", output.display()))?;
    let mut writer = BufWriter::new(file);
    data.write(&mut writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(feature = "cpu_onnx-convert")]
fn compile_onnx(model: rten_onnx::onnx::ModelProto, input_shape: &[usize]) -> Result<ModelData> {
    use rten_onnx::onnx::DataType;
    use std::collections::HashMap;

    let graph = model.graph.context("ONNX model has no graph")?;
    ensure!(graph.output.len() == 1, "expected one ONNX graph output");
    let mut names = HashMap::<String, ValueId>::new();
    let mut initial_values = Vec::<Option<Tensor>>::new();
    for initializer in graph.initializer {
        let name = initializer
            .name
            .clone()
            .context("ONNX initializer has no name")?;
        ensure!(!names.contains_key(&name), "duplicate ONNX value {name:?}");
        let shape = initializer
            .dims
            .iter()
            .map(|&dimension| {
                ensure!(
                    dimension >= 0,
                    "initializer {name:?} has a negative dimension"
                );
                usize::try_from(dimension).map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            initializer.external_data.is_empty(),
            "external ONNX weights are unsupported"
        );
        let expected = element_count(&shape).context("initializer shape overflow")?;
        let tensor = if initializer.data_type == Some(DataType::FLOAT) {
            let values = if let Some(raw) = &initializer.raw_data {
                let raw = raw.borrow();
                ensure!(
                    raw.len() == expected * 4,
                    "initializer {name:?} byte length mismatch"
                );
                raw.chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                    .collect()
            } else {
                ensure!(
                    initializer.float_data.len() == expected,
                    "initializer {name:?} length mismatch"
                );
                initializer.float_data
            };
            Tensor::new_f32(shape, values)
        } else if initializer.data_type == Some(DataType::INT64) {
            let values = if let Some(raw) = &initializer.raw_data {
                let raw = raw.borrow();
                ensure!(
                    raw.len() == expected * 8,
                    "initializer {name:?} byte length mismatch"
                );
                raw.chunks_exact(8)
                    .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("eight-byte chunk")))
                    .collect()
            } else {
                ensure!(
                    initializer.int64_data.len() == expected,
                    "initializer {name:?} length mismatch"
                );
                initializer.int64_data
            };
            Tensor::new_i64(shape, values)
        } else {
            bail!(
                "initializer {name:?} has unsupported data type {:?}",
                initializer.data_type
            );
        };
        let id = initial_values.len();
        names.insert(name, id);
        initial_values.push(Some(tensor));
    }

    let graph_inputs = graph
        .input
        .into_iter()
        .filter_map(|input| input.name)
        .filter(|name| !names.contains_key(name))
        .collect::<Vec<_>>();
    ensure!(
        graph_inputs.len() == 1,
        "expected one non-initializer ONNX graph input"
    );
    let input = initial_values.len();
    names.insert(graph_inputs[0].clone(), input);
    initial_values.push(None);

    let mut nodes = Vec::new();
    for (index, proto) in graph.node.into_iter().enumerate() {
        let op_type = proto
            .op_type
            .as_deref()
            .context("ONNX node has no operation type")?;
        ensure!(
            proto.output.len() == 1,
            "{op_type} node must have one output"
        );
        if op_type == "Identity" {
            ensure!(proto.input.len() == 1, "Identity node must have one input");
            let input_id = *names.get(&proto.input[0]).with_context(|| {
                format!("Identity references unknown value {:?}", proto.input[0])
            })?;
            names.insert(proto.output[0].clone(), input_id);
            continue;
        }
        let inputs = proto
            .input
            .iter()
            .map(|name| {
                if name.is_empty() {
                    Ok(None)
                } else {
                    names
                        .get(name)
                        .copied()
                        .map(Some)
                        .with_context(|| format!("{op_type} references unknown value {name:?}"))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let operation = operation_from_onnx(op_type, &proto.attribute)?;
        let output = initial_values.len();
        ensure!(
            names.insert(proto.output[0].clone(), output).is_none(),
            "duplicate ONNX value {:?}",
            proto.output[0]
        );
        initial_values.push(None);
        nodes.push(Node {
            name: proto.name.unwrap_or_else(|| format!("{op_type}.{index}")),
            inputs,
            output,
            operation,
        });
    }
    let output_name = graph.output[0]
        .name
        .as_ref()
        .context("ONNX graph output has no name")?;
    let output = *names
        .get(output_name)
        .with_context(|| format!("unknown ONNX graph output {output_name:?}"))?;
    Ok(ModelData {
        input,
        output,
        input_shape: input_shape.to_vec(),
        initial_values,
        nodes,
        use_counts: Vec::new(),
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn operation_from_onnx(
    op_type: &str,
    attributes: &[rten_onnx::onnx::AttributeProto],
) -> Result<Operation> {
    let operation = match op_type {
        "Add" => Operation::Add,
        "AveragePool" => Operation::AveragePool(pool_options(attributes)?),
        "BatchNormalization" => Operation::BatchNormalization {
            epsilon: attribute_float(attributes, "epsilon").unwrap_or(1e-5),
        },
        "Concat" => Operation::Concat {
            axis: attribute_int(attributes, "axis").context("Concat has no axis")?,
        },
        "Conv" => Operation::Conv(conv_options(attributes)?),
        "ConvTranspose" => Operation::ConvTranspose(conv_options(attributes)?),
        "Div" => Operation::Div,
        "Erf" => Operation::Erf,
        "GlobalAveragePool" => Operation::GlobalAveragePool,
        "HardSigmoid" => Operation::HardSigmoid {
            alpha: attribute_float(attributes, "alpha").unwrap_or(0.2),
            beta: attribute_float(attributes, "beta").unwrap_or(0.5),
        },
        "MatMul" => Operation::MatMul,
        "MaxPool" => Operation::MaxPool(pool_options(attributes)?),
        "Mul" => Operation::Mul,
        "Pow" => Operation::Pow,
        "ReduceMean" => Operation::ReduceMean {
            axes: attribute_ints(attributes, "axes")
                .context("ReduceMean has no axes")?
                .to_vec(),
            keep_dims: attribute_int(attributes, "keepdims").unwrap_or(1) != 0,
        },
        "Relu" => Operation::Relu,
        "Reshape" => Operation::Reshape,
        "Resize" => {
            ensure!(
                attribute_string(attributes, "mode").unwrap_or("nearest") == "nearest",
                "only nearest Resize is supported"
            );
            ensure!(
                attribute_string(attributes, "coordinate_transformation_mode")
                    .unwrap_or("half_pixel")
                    == "asymmetric",
                "only asymmetric Resize is supported"
            );
            Operation::Resize
        }
        "Shape" => Operation::Shape,
        "Sigmoid" => Operation::Sigmoid,
        "Slice" => Operation::Slice,
        "Softmax" => Operation::Softmax {
            axis: attribute_int(attributes, "axis").unwrap_or(-1),
        },
        "Sqrt" => Operation::Sqrt,
        "Squeeze" => Operation::Squeeze {
            axes: attribute_ints(attributes, "axes").unwrap_or(&[]).to_vec(),
        },
        "Sub" => Operation::Sub,
        "Transpose" => Operation::Transpose {
            permutation: attribute_ints(attributes, "perm")
                .context("Transpose has no permutation")?
                .iter()
                .map(|&value| usize::try_from(value).map_err(anyhow::Error::from))
                .collect::<Result<Vec<_>>>()?,
        },
        "Unsqueeze" => Operation::Unsqueeze {
            axes: attribute_ints(attributes, "axes").unwrap_or(&[]).to_vec(),
        },
        other => bail!("unsupported ONNX operation {other}"),
    };
    Ok(operation)
}

#[cfg(feature = "cpu_onnx-convert")]
fn conv_options(attributes: &[rten_onnx::onnx::AttributeProto]) -> Result<ConvOptions> {
    let strides = pair(
        attribute_ints(attributes, "strides").unwrap_or(&[1, 1]),
        "strides",
    )?;
    let kernel = pair(
        attribute_ints(attributes, "kernel_shape").context("Conv has no kernel shape")?,
        "kernel shape",
    )?;
    let pads = padding(attributes, kernel, strides)?;
    ensure!(
        attribute_ints(attributes, "dilations").unwrap_or(&[1, 1]) == [1, 1],
        "dilated Conv is unsupported"
    );
    Ok(ConvOptions {
        strides,
        pads,
        groups: usize::try_from(attribute_int(attributes, "group").unwrap_or(1))?,
        packed_pointwise: false,
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn pool_options(attributes: &[rten_onnx::onnx::AttributeProto]) -> Result<PoolOptions> {
    let kernel = pair(
        attribute_ints(attributes, "kernel_shape").context("pool has no kernel shape")?,
        "kernel shape",
    )?;
    let strides = pair(
        attribute_ints(attributes, "strides").unwrap_or(&[1, 1]),
        "strides",
    )?;
    Ok(PoolOptions {
        kernel,
        strides,
        pads: padding(attributes, kernel, strides)?,
        ceil_mode: attribute_int(attributes, "ceil_mode").unwrap_or(0) != 0,
        count_include_pad: attribute_int(attributes, "count_include_pad").unwrap_or(0) != 0,
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn padding(
    attributes: &[rten_onnx::onnx::AttributeProto],
    kernel: [usize; 2],
    strides: [usize; 2],
) -> Result<[usize; 4]> {
    if let Some(pads) = attribute_ints(attributes, "pads") {
        ensure!(pads.len() == 4, "padding must contain four values");
        return pads
            .iter()
            .map(|&value| usize::try_from(value).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("padding must contain four values"));
    }
    match attribute_string(attributes, "auto_pad").unwrap_or("NOTSET") {
        "NOTSET" | "VALID" => Ok([0; 4]),
        "SAME_UPPER" => {
            ensure!(
                strides == [1, 1],
                "SAME_UPPER with non-unit stride is unsupported"
            );
            Ok([0, 0, kernel[0] - 1, kernel[1] - 1])
        }
        value => bail!("unsupported automatic padding mode {value:?}"),
    }
}

#[cfg(feature = "cpu_onnx-convert")]
fn pair(values: &[i64], name: &str) -> Result<[usize; 2]> {
    ensure!(values.len() == 2, "{name} must contain two values");
    Ok([usize::try_from(values[0])?, usize::try_from(values[1])?])
}

#[cfg(feature = "cpu_onnx-convert")]
fn attribute<'a>(
    attributes: &'a [rten_onnx::onnx::AttributeProto],
    name: &str,
) -> Option<&'a rten_onnx::onnx::AttributeProto> {
    attributes
        .iter()
        .find(|attribute| attribute.name.as_deref() == Some(name))
}

#[cfg(feature = "cpu_onnx-convert")]
fn attribute_int(attributes: &[rten_onnx::onnx::AttributeProto], name: &str) -> Option<i64> {
    attribute(attributes, name).and_then(|attribute| attribute.i)
}

#[cfg(feature = "cpu_onnx-convert")]
fn attribute_float(attributes: &[rten_onnx::onnx::AttributeProto], name: &str) -> Option<f32> {
    attribute(attributes, name).and_then(|attribute| attribute.f)
}

#[cfg(feature = "cpu_onnx-convert")]
fn attribute_ints<'a>(
    attributes: &'a [rten_onnx::onnx::AttributeProto],
    name: &str,
) -> Option<&'a [i64]> {
    attribute(attributes, name).map(|attribute| attribute.ints.as_slice())
}

#[cfg(feature = "cpu_onnx-convert")]
fn attribute_string<'a>(
    attributes: &'a [rten_onnx::onnx::AttributeProto],
    name: &str,
) -> Option<&'a str> {
    attribute(attributes, name).and_then(|attribute| attribute.s.as_deref())
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_operation(writer: &mut impl Write, operation: &Operation) -> Result<()> {
    let tag = match operation {
        Operation::Add => 0,
        Operation::AveragePool(_) => 1,
        Operation::BatchNormalization { .. } => 2,
        Operation::Concat { .. } => 3,
        Operation::Conv(_) => 4,
        Operation::ConvTranspose(_) => 5,
        Operation::Div => 6,
        Operation::Erf => 7,
        Operation::GlobalAveragePool => 8,
        Operation::HardSigmoid { .. } => 9,
        Operation::MatMul => 10,
        Operation::MaxPool(_) => 11,
        Operation::Mul => 12,
        Operation::Pow => 13,
        Operation::ReduceMean { .. } => 14,
        Operation::Relu => 15,
        Operation::Reshape => 16,
        Operation::Resize => 17,
        Operation::Shape => 18,
        Operation::Sigmoid => 19,
        Operation::Slice => 20,
        Operation::Softmax { .. } => 21,
        Operation::Sqrt => 22,
        Operation::Squeeze { .. } => 23,
        Operation::Sub => 24,
        Operation::Transpose { .. } => 25,
        Operation::Unsqueeze { .. } => 26,
        Operation::Gelu => 27,
        Operation::HardSwish => 28,
        Operation::Silu => 29,
        Operation::ConvGelu(_) => 30,
        Operation::BiasSoftmax { .. } => 31,
        Operation::MatMulBiasSoftmax { .. } => 32,
    };
    writer.write_all(&[tag])?;
    match operation {
        Operation::AveragePool(options) | Operation::MaxPool(options) => {
            write_pool(writer, options)?
        }
        Operation::BatchNormalization { epsilon } => writer.write_all(&epsilon.to_le_bytes())?,
        Operation::BiasSoftmax { axis }
        | Operation::Concat { axis }
        | Operation::MatMulBiasSoftmax { axis }
        | Operation::Softmax { axis } => writer.write_all(&axis.to_le_bytes())?,
        Operation::Conv(options)
        | Operation::ConvGelu(options)
        | Operation::ConvTranspose(options) => write_conv(writer, options)?,
        Operation::HardSigmoid { alpha, beta } => {
            writer.write_all(&alpha.to_le_bytes())?;
            writer.write_all(&beta.to_le_bytes())?;
        }
        Operation::ReduceMean { axes, keep_dims } => {
            write_i64_vec(writer, axes)?;
            writer.write_all(&[u8::from(*keep_dims)])?;
        }
        Operation::Squeeze { axes } | Operation::Unsqueeze { axes } => write_i64_vec(writer, axes)?,
        Operation::Transpose { permutation } => write_usize_vec(writer, permutation)?,
        _ => {}
    }
    Ok(())
}

fn read_operation(reader: &mut impl Read) -> Result<Operation> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    Ok(match tag[0] {
        0 => Operation::Add,
        1 => Operation::AveragePool(read_pool(reader)?),
        2 => Operation::BatchNormalization {
            epsilon: read_f32(reader)?,
        },
        3 => Operation::Concat {
            axis: read_i64(reader)?,
        },
        4 => Operation::Conv(read_conv(reader)?),
        5 => Operation::ConvTranspose(read_conv(reader)?),
        6 => Operation::Div,
        7 => Operation::Erf,
        8 => Operation::GlobalAveragePool,
        9 => Operation::HardSigmoid {
            alpha: read_f32(reader)?,
            beta: read_f32(reader)?,
        },
        10 => Operation::MatMul,
        11 => Operation::MaxPool(read_pool(reader)?),
        12 => Operation::Mul,
        13 => Operation::Pow,
        14 => {
            let axes = read_i64_vec(reader, MAX_RANK, "reduction axes")?;
            let mut keep_dims = [0u8; 1];
            reader.read_exact(&mut keep_dims)?;
            ensure!(keep_dims[0] <= 1, "invalid keep-dimensions flag");
            Operation::ReduceMean {
                axes,
                keep_dims: keep_dims[0] != 0,
            }
        }
        15 => Operation::Relu,
        16 => Operation::Reshape,
        17 => Operation::Resize,
        18 => Operation::Shape,
        19 => Operation::Sigmoid,
        20 => Operation::Slice,
        21 => Operation::Softmax {
            axis: read_i64(reader)?,
        },
        22 => Operation::Sqrt,
        23 => Operation::Squeeze {
            axes: read_i64_vec(reader, MAX_RANK, "squeeze axes")?,
        },
        24 => Operation::Sub,
        25 => Operation::Transpose {
            permutation: read_usize_vec(reader, MAX_RANK, "transpose permutation")?,
        },
        26 => Operation::Unsqueeze {
            axes: read_i64_vec(reader, MAX_RANK, "unsqueeze axes")?,
        },
        27 => Operation::Gelu,
        28 => Operation::HardSwish,
        29 => Operation::Silu,
        30 => Operation::ConvGelu(read_conv(reader)?),
        31 => Operation::BiasSoftmax {
            axis: read_i64(reader)?,
        },
        32 => Operation::MatMulBiasSoftmax {
            axis: read_i64(reader)?,
        },
        value => bail!("unknown operation tag {value}"),
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_conv(writer: &mut impl Write, options: &ConvOptions) -> Result<()> {
    write_usize_vec(writer, &options.strides)?;
    write_usize_vec(writer, &options.pads)?;
    write_u32(writer, to_u32(options.groups, "group count")?)
}

fn read_conv(reader: &mut impl Read) -> Result<ConvOptions> {
    Ok(ConvOptions {
        strides: read_usize_vec(reader, 2, "convolution strides")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("convolution must have two strides"))?,
        pads: read_usize_vec(reader, 4, "convolution padding")?
            .try_into()
            .map_err(|_| anyhow::anyhow!("convolution must have four padding values"))?,
        groups: usize::try_from(read_u32(reader)?)?,
        packed_pointwise: false,
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_pool(writer: &mut impl Write, options: &PoolOptions) -> Result<()> {
    write_usize_vec(writer, &options.kernel)?;
    write_usize_vec(writer, &options.strides)?;
    write_usize_vec(writer, &options.pads)?;
    writer.write_all(&[
        u8::from(options.ceil_mode),
        u8::from(options.count_include_pad),
    ])?;
    Ok(())
}

fn read_pool(reader: &mut impl Read) -> Result<PoolOptions> {
    let kernel = read_usize_vec(reader, 2, "pool kernel")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("pool must have a two-dimensional kernel"))?;
    let strides = read_usize_vec(reader, 2, "pool strides")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("pool must have two strides"))?;
    let pads = read_usize_vec(reader, 4, "pool padding")?
        .try_into()
        .map_err(|_| anyhow::anyhow!("pool must have four padding values"))?;
    let mut flags = [0u8; 2];
    reader.read_exact(&mut flags)?;
    ensure!(flags.iter().all(|&flag| flag <= 1), "invalid pool flags");
    Ok(PoolOptions {
        kernel,
        strides,
        pads,
        ceil_mode: flags[0] != 0,
        count_include_pad: flags[1] != 0,
    })
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_usize_vec(writer: &mut impl Write, values: &[usize]) -> Result<()> {
    write_u32(writer, to_u32(values.len(), "vector length")?)?;
    for &value in values {
        write_u64(writer, value as u64)?;
    }
    Ok(())
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_i64_vec(writer: &mut impl Write, values: &[i64]) -> Result<()> {
    write_u32(writer, to_u32(values.len(), "vector length")?)?;
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn read_usize_vec(reader: &mut impl Read, maximum: usize, name: &str) -> Result<Vec<usize>> {
    let length = read_len(reader, maximum, name)?;
    (0..length)
        .map(|_| Ok(usize::try_from(read_u64(reader)?)?))
        .collect()
}

fn read_i64_vec(reader: &mut impl Read, maximum: usize, name: &str) -> Result<Vec<i64>> {
    let length = read_len(reader, maximum, name)?;
    (0..length).map(|_| read_i64(reader)).collect()
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    write_u32(writer, to_u32(value.len(), "string length")?)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let length = read_len(reader, 64 * 1024, "string length")?;
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("model contains invalid UTF-8")
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_u32(writer: &mut impl Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

#[cfg(feature = "cpu_onnx-convert")]
fn write_u64(writer: &mut impl Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn read_u32(reader: &mut impl Read) -> Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(reader: &mut impl Read) -> Result<i64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_len(reader: &mut impl Read, maximum: usize, name: &str) -> Result<usize> {
    let value = usize::try_from(read_u32(reader)?)?;
    ensure!(value <= maximum, "{name} {value} exceeds limit {maximum}");
    Ok(value)
}

fn read_id(reader: &mut impl Read, value_count: usize, name: &str) -> Result<usize> {
    let id = usize::try_from(read_u32(reader)?)?;
    ensure!(id < value_count, "{name} {id} is out of range");
    Ok(id)
}

#[cfg(feature = "cpu_onnx-convert")]
fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} does not fit u32"))
}
