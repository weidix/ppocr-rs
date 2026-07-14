use super::{
    kernels,
    ops::{ConvOptions, Node, Operation, PoolOptions, SparseConvWeights},
    tensor::{IntoShape, Tensor, element_count},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

fn run(operation: Operation, inputs: Vec<Tensor>) -> Result<Tensor> {
    Node {
        name: String::new(),
        operation,
    }
    .run(inputs)
}

impl Tensor {
    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn dim(&self, axis: usize) -> Result<usize> {
        self.shape
            .get(axis)
            .copied()
            .with_context(|| format!("axis {axis} is out of range for shape {:?}", self.shape))
    }

    pub(crate) fn dims2(&self) -> Result<(usize, usize)> {
        let [first, second] = self
            .shape
            .as_slice()
            .try_into()
            .with_context(|| format!("expected rank two, found shape {:?}", self.shape))?;
        Ok((first, second))
    }

    pub(crate) fn dims3(&self) -> Result<(usize, usize, usize)> {
        let [first, second, third] = self
            .shape
            .as_slice()
            .try_into()
            .with_context(|| format!("expected rank three, found shape {:?}", self.shape))?;
        Ok((first, second, third))
    }

    pub(crate) fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let [first, second, third, fourth] = self
            .shape
            .as_slice()
            .try_into()
            .with_context(|| format!("expected rank four, found shape {:?}", self.shape))?;
        Ok((first, second, third, fourth))
    }

    pub(crate) fn reshape(&self, shape: impl IntoShape) -> Result<Self> {
        let shape = shape
            .into_shape()
            .into_iter()
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        run(
            Operation::Reshape,
            vec![self.clone(), Tensor::new_i64(vec![shape.len()], shape)],
        )
    }

    pub(crate) fn flatten(&self, start: usize, end: usize) -> Result<Self> {
        ensure!(
            start <= end && end < self.rank(),
            "invalid flatten range {start}..={end} for shape {:?}",
            self.shape
        );
        let flattened =
            element_count(&self.shape[start..=end]).context("flatten shape overflow")?;
        let mut shape = Vec::with_capacity(self.rank() - (end - start));
        shape.extend_from_slice(&self.shape[..start]);
        shape.push(flattened);
        shape.extend_from_slice(&self.shape[end + 1..]);
        self.reshape(shape)
    }

    pub(crate) fn transpose(&self, first: usize, second: usize) -> Result<Self> {
        ensure!(
            first < self.rank() && second < self.rank(),
            "transpose axes ({first}, {second}) are out of range for shape {:?}",
            self.shape
        );
        let mut permutation = (0..self.rank()).collect::<Vec<_>>();
        permutation.swap(first, second);
        self.permute(permutation)
    }

    pub(crate) fn permute(&self, permutation: impl IntoShape) -> Result<Self> {
        run(
            Operation::Transpose {
                permutation: permutation.into_shape(),
            },
            vec![self.clone()],
        )
    }

    pub(crate) fn squeeze(&self, axis: usize) -> Result<Self> {
        run(
            Operation::Squeeze {
                axes: vec![i64::try_from(axis)?],
            },
            vec![self.clone()],
        )
    }

    pub(crate) fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self> {
        let dimension = self.dim(axis)?;
        let end = start.checked_add(length).context("narrow range overflow")?;
        ensure!(
            end <= dimension,
            "narrow range {start}..{end} exceeds dimension {dimension}"
        );
        run(
            Operation::Slice,
            vec![
                self.clone(),
                Tensor::new_i64(vec![1], vec![i64::try_from(start)?]),
                Tensor::new_i64(vec![1], vec![i64::try_from(end)?]),
                Tensor::new_i64(vec![1], vec![i64::try_from(axis)?]),
            ],
        )
    }

    pub(crate) fn chunk(&self, chunks: usize, axis: usize) -> Result<Vec<Self>> {
        ensure!(chunks > 0, "chunk count must be positive");
        let dimension = self.dim(axis)?;
        ensure!(
            dimension.is_multiple_of(chunks),
            "dimension {dimension} is not divisible by {chunks} chunks"
        );
        let chunk = dimension / chunks;
        (0..chunks)
            .map(|index| self.narrow(axis, index * chunk, chunk))
            .collect()
    }

    pub(crate) fn cat(inputs: &[&Self], axis: usize) -> Result<Self> {
        ensure!(
            !inputs.is_empty(),
            "cannot concatenate an empty tensor list"
        );
        run(
            Operation::Concat {
                axis: i64::try_from(axis)?,
            },
            inputs.iter().map(|input| (*input).clone()).collect(),
        )
    }

    pub(crate) fn add(&self, other: &Self) -> Result<Self> {
        run(Operation::Add, vec![self.clone(), other.clone()])
    }

    pub(crate) fn into_add(self, other: &Self) -> Result<Self> {
        run(Operation::Add, vec![self, other.clone()])
    }

    pub(crate) fn into_mul(self, other: &Self) -> Result<Self> {
        run(Operation::Mul, vec![self, other.clone()])
    }

    pub(crate) fn into_residual_mul(self, gate: &Self) -> Result<Self> {
        let (batch, channels, height, width) = self.dims4()?;
        ensure!(
            gate.dims4()? == (batch, channels, 1, 1),
            "residual gate shape {:?} does not match input shape {:?}",
            gate.shape,
            self.shape
        );
        let plane = height
            .checked_mul(width)
            .context("residual feature plane overflow")?;
        let gate = gate.as_f32()?;
        let mut output = self;
        output
            .f32_mut()?
            .chunks_mut(plane)
            .zip(gate.iter())
            .for_each(|(values, gate)| kernels::residual_mul_in_place(values, *gate));
        Ok(output)
    }

    pub(crate) fn into_affine(self, scale: f32, bias: f32) -> Result<Self> {
        let mut output = self;
        kernels::affine_in_place(output.f32_mut()?, scale, bias);
        Ok(output)
    }

    pub(crate) fn into_relu(self) -> Result<Self> {
        run(Operation::Relu, vec![self])
    }

    pub(crate) fn into_silu(self) -> Result<Self> {
        run(Operation::Silu, vec![self])
    }

    pub(crate) fn into_sigmoid(self) -> Result<Self> {
        run(Operation::Sigmoid, vec![self])
    }

    pub(crate) fn into_hard_sigmoid(self, alpha: f32, beta: f32) -> Result<Self> {
        run(Operation::HardSigmoid { alpha, beta }, vec![self])
    }

    pub(crate) fn into_hard_swish(self) -> Result<Self> {
        run(Operation::HardSwish, vec![self])
    }

    pub(crate) fn matmul(&self, other: &Self) -> Result<Self> {
        run(Operation::MatMul, vec![self.clone(), other.clone()])
    }

    pub(crate) fn into_softmax(self, axis: i64) -> Result<Self> {
        run(Operation::Softmax { axis }, vec![self])
    }

    pub(crate) fn max_pool2d(
        &self,
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        ceil_mode: bool,
    ) -> Result<Self> {
        run(
            Operation::MaxPool(PoolOptions {
                kernel,
                strides,
                pads,
                ceil_mode,
                count_include_pad: false,
            }),
            vec![self.clone()],
        )
    }

    pub(crate) fn avg_pool2d(
        &self,
        kernel: [usize; 2],
        strides: [usize; 2],
        pads: [usize; 4],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Self> {
        run(
            Operation::AveragePool(PoolOptions {
                kernel,
                strides,
                pads,
                ceil_mode,
                count_include_pad,
            }),
            vec![self.clone()],
        )
    }

    pub(crate) fn global_avg_pool2d(&self) -> Result<Self> {
        run(Operation::GlobalAveragePool, vec![self.clone()])
    }

    pub(crate) fn resize_nearest2d(&self, size: [usize; 2]) -> Result<Self> {
        let (batch, channels, _, _) = self.dims4()?;
        let shape = [batch, channels, size[0], size[1]]
            .into_iter()
            .map(i64::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        run(
            Operation::Resize,
            vec![
                self.clone(),
                Tensor::new_f32(vec![0], Vec::new()),
                Tensor::new_i64(vec![4], shape),
            ],
        )
    }
}

#[derive(Clone)]
pub(crate) struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    options: ConvOptions,
}

impl Conv2d {
    pub(crate) fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        strides: [usize; 2],
        pads: [usize; 4],
        groups: usize,
    ) -> Result<Self> {
        Self::new_with_pointwise_prune_threshold(weight, bias, strides, pads, groups, 4.0e-2)
    }

    pub(crate) fn new_with_pointwise_prune_threshold(
        weight: Tensor,
        bias: Option<Tensor>,
        strides: [usize; 2],
        pads: [usize; 4],
        groups: usize,
        pointwise_prune_threshold: f32,
    ) -> Result<Self> {
        let [
            output_channels,
            channels_per_group,
            kernel_height,
            kernel_width,
        ]: [usize; 4] =
            weight.shape.as_slice().try_into().with_context(|| {
                format!("expected rank-four Conv weight, found {:?}", weight.shape)
            })?;
        weight.as_f32()?;
        ensure!(groups > 0, "Conv group count must be positive");
        ensure!(
            pointwise_prune_threshold.is_finite() && pointwise_prune_threshold >= 0.0,
            "pointwise prune threshold must be finite and non-negative"
        );
        ensure!(
            strides.into_iter().all(|stride| stride > 0),
            "Conv strides must be positive"
        );
        if let Some(bias) = &bias {
            ensure!(
                bias.as_f32()?.len() == output_channels,
                "Conv bias length does not match output channels"
            );
        }

        let inner = channels_per_group * kernel_height * kernel_width;
        let tiled_spatial = groups == 1
            && (kernel_height != 1 || kernel_width != 1)
            && inner >= 128
            && output_channels >= 16;
        let sparse_pointwise = groups == 1
            && kernel_height == 1
            && kernel_width == 1
            && inner >= 512
            && output_channels >= 512;
        // Projection weights are strongly block-pruned. The tighter spatial
        // threshold preserves detector scores; large pointwise layers use the
        // OCR decision-preserving threshold measured by the CPU benchmark.
        let prune_threshold = if tiled_spatial {
            2.0e-6
        } else {
            pointwise_prune_threshold
        };
        let sparse_weights = ((tiled_spatial || sparse_pointwise)
            && output_channels.is_multiple_of(4))
        .then(|| {
            SparseConvWeights::from_dense(
                weight.as_f32().expect("Conv weight was validated as F32"),
                output_channels,
                inner,
                prune_threshold,
            )
        })
        .flatten();
        let packed_pointwise = groups == 1
            && ((kernel_height == 1 && kernel_width == 1)
                || (strides != [1, 1] && output_channels >= 48)
                || tiled_spatial);
        let weight = if packed_pointwise {
            if tiled_spatial || sparse_weights.is_some() {
                pack_conv_rows(weight, output_channels, 4)?
            } else {
                pack_conv_rows(weight, output_channels, 12)?
            }
        } else {
            weight
        };
        Ok(Self {
            weight,
            bias,
            options: ConvOptions {
                strides,
                pads,
                groups,
                packed_pointwise,
                sparse_weights,
            },
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, false)
    }

    pub(crate) fn forward_gelu(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, true)
    }

    fn run(&self, input: &Tensor, gelu: bool) -> Result<Tensor> {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(bias) = &self.bias {
            inputs.push(bias.clone());
        }
        run(
            if gelu {
                Operation::ConvGelu(self.options.clone())
            } else {
                Operation::Conv(self.options.clone())
            },
            inputs,
        )
    }
}

fn pack_conv_rows(weight: Tensor, rows: usize, block_rows: usize) -> Result<Tensor> {
    ensure!(rows > 0, "Conv weight has zero output channels");
    ensure!(block_rows > 0, "Conv weight block has zero rows");
    let shape = weight.shape.clone();
    let source = weight.into_f32()?;
    ensure!(
        source.len().is_multiple_of(rows),
        "Conv weight size is not divisible by output channels"
    );
    let inner = source.len() / rows;
    let mut packed = Vec::with_capacity(source.len());
    for row_start in (0..rows).step_by(block_rows) {
        let block_rows = (rows - row_start).min(block_rows);
        for index in 0..inner {
            for row in 0..block_rows {
                packed.push(source[(row_start + row) * inner + index]);
            }
        }
    }
    Ok(Tensor::new_f32(shape, packed))
}

#[derive(Clone)]
pub(crate) struct ConvTranspose2d {
    weight: Tensor,
    bias: Option<Tensor>,
    options: ConvOptions,
}

impl ConvTranspose2d {
    pub(crate) fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        strides: [usize; 2],
        pads: [usize; 4],
        groups: usize,
    ) -> Result<Self> {
        let [input_channels, output_channels_per_group, _, _]: [usize; 4] =
            weight.shape.as_slice().try_into().with_context(|| {
                format!(
                    "expected rank-four ConvTranspose weight, found {:?}",
                    weight.shape
                )
            })?;
        weight.as_f32()?;
        ensure!(groups > 0, "ConvTranspose group count must be positive");
        ensure!(
            input_channels.is_multiple_of(groups),
            "ConvTranspose input channels are not divisible by groups"
        );
        ensure!(
            strides.into_iter().all(|stride| stride > 0),
            "ConvTranspose strides must be positive"
        );
        if let Some(bias) = &bias {
            ensure!(
                bias.as_f32()?.len() == output_channels_per_group * groups,
                "ConvTranspose bias length does not match output channels"
            );
        }
        Ok(Self {
            weight,
            bias,
            options: ConvOptions {
                strides,
                pads,
                groups,
                packed_pointwise: false,
                sparse_weights: None,
            },
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(bias) = &self.bias {
            inputs.push(bias.clone());
        }
        run(Operation::ConvTranspose(self.options.clone()), inputs)
    }
}

#[derive(Clone)]
pub(crate) struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    epsilon: f32,
}

impl LayerNorm {
    pub(crate) fn new(weight: Tensor, bias: Tensor, epsilon: f32) -> Result<Self> {
        let features = weight.as_f32()?.len();
        ensure!(features > 0, "LayerNorm must have at least one feature");
        ensure!(
            weight.rank() == 1 && bias.rank() == 1 && bias.as_f32()?.len() == features,
            "LayerNorm weight and bias must be equal-length vectors"
        );
        ensure!(epsilon >= 0.0, "LayerNorm epsilon must be non-negative");
        Ok(Self {
            weight,
            bias,
            epsilon,
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.weight.len();
        ensure!(
            input.shape.last() == Some(&features),
            "LayerNorm feature count does not match input shape {:?}",
            input.shape
        );
        let weight = self.weight.as_f32()?;
        let bias = self.bias.as_f32()?;
        let mut output = input.as_f32()?.to_vec();
        output.par_chunks_mut(features).for_each(|row| {
            let mean = kernels::mean(row);
            let variance = row
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / features as f32;
            let inverse_std = (variance + self.epsilon).sqrt().recip();
            for ((value, weight), bias) in row.iter_mut().zip(weight).zip(bias) {
                *value = (*value - mean).mul_add(inverse_std * *weight, *bias);
            }
        });
        Ok(Tensor::new_f32(input.shape.clone(), output))
    }
}

#[derive(Clone)]
pub(crate) struct Linear {
    // Safetensors stores Linear weights as [out, in]. Keep that orientation and
    // interleave blocks of output rows for the packed-left GEMM kernels.
    packed_weight: Tensor,
    input_features: usize,
    output_features: usize,
    bias: Option<Tensor>,
}

impl Linear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        ensure!(
            weight.rank() == 2,
            "Linear weight must have rank two, found {:?}",
            weight.shape
        );
        let (output_features, input_features) = weight.dims2()?;
        ensure!(
            input_features > 0 && output_features > 0,
            "Linear feature counts must be positive"
        );
        if let Some(bias) = &bias {
            ensure!(
                bias.rank() == 1 && bias.as_f32()?.len() == output_features,
                "Linear bias length does not match output features"
            );
        }
        let packed_weight = pack_conv_rows(weight, output_features, 12)?;
        Ok(Self {
            packed_weight,
            input_features,
            output_features,
            bias,
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, false)
    }

    pub(crate) fn forward_softmax(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, true)
    }

    fn run(&self, input: &Tensor, apply_softmax: bool) -> Result<Tensor> {
        ensure!(
            !input.shape.is_empty(),
            "Linear input must have at least one dimension"
        );
        ensure!(
            input.shape.last() == Some(&self.input_features),
            "Linear input feature count does not match input shape {:?}",
            input.shape
        );
        let rows = element_count(&input.shape[..input.shape.len() - 1])
            .context("Linear input shape overflow")?;
        let output_len = rows
            .checked_mul(self.output_features)
            .context("Linear output shape overflow")?;
        let mut output_shape = input.shape.clone();
        *output_shape.last_mut().expect("non-empty Linear shape") = self.output_features;
        if output_len == 0 {
            return Ok(Tensor::new_f32(output_shape, Vec::new()));
        }

        let transposed_input = transpose_matrix(input.as_f32()?, rows, self.input_features);
        let mut transposed_output = vec![0.0; output_len];
        kernels::gemm_packed_left(
            &mut transposed_output,
            self.packed_weight.as_f32()?,
            &transposed_input,
            self.output_features,
            self.input_features,
            rows,
            self.bias.as_ref().map(Tensor::as_f32).transpose()?,
        );
        let mut output = transpose_matrix(&transposed_output, self.output_features, rows);
        if apply_softmax {
            output
                .par_chunks_mut(self.output_features)
                .for_each(kernels::softmax_in_place);
        }
        Ok(Tensor::new_f32(output_shape, output))
    }
}

fn transpose_matrix(input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    assert_eq!(rows.checked_mul(columns), Some(input.len()));
    let mut output = vec![0.0; input.len()];
    output
        .par_chunks_mut(rows)
        .enumerate()
        .for_each(|(column, output_row)| {
            for (row, output) in output_row.iter_mut().enumerate() {
                *output = input[row * columns + column];
            }
        });
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: impl Into<Vec<usize>>, values: &[f32]) -> Tensor {
        Tensor::from_f32(shape, values.to_vec()).unwrap()
    }

    #[test]
    fn shape_conversions_cover_model_shapes() {
        assert_eq!(3usize.into_shape(), [3]);
        assert_eq!((2, 3, 4, 5, 6).into_shape(), [2, 3, 4, 5, 6]);
        assert_eq!([7, 8].into_shape(), [7, 8]);
    }

    #[test]
    fn tensor_shape_ops_keep_row_major_order() {
        let input = tensor([1, 2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let transposed = input.transpose(1, 2).unwrap();
        assert_eq!(transposed.shape(), [1, 3, 2]);
        assert_eq!(
            transposed.as_f32().unwrap(),
            &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
        assert_eq!(transposed.reshape((3, 2)).unwrap().shape(), [3, 2]);
        assert!(transposed.reshape((4, 2)).is_err());
    }

    #[test]
    fn linear_transposes_weights_once_and_fuses_softmax() {
        let weight = tensor([2, 3], &[1.0, 0.0, -1.0, 0.0, 1.0, 1.0]);
        let bias = tensor([2], &[0.5, -0.5]);
        let linear = Linear::new(weight, Some(bias)).unwrap();
        let input = tensor([1, 3], &[2.0, 3.0, 1.0]);
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.shape(), [1, 2]);
        assert_eq!(output.as_f32().unwrap(), &[1.5, 3.5]);

        let probabilities = linear.forward_softmax(&input).unwrap();
        let values = probabilities.as_f32().unwrap();
        assert!((values[0] + values[1] - 1.0).abs() < 1e-6);
        assert!(values[1] > values[0]);
    }

    #[test]
    fn packed_linear_matches_dynamic_matmul_with_leading_dimensions() {
        let weight = tensor(
            [4, 3],
            &[
                1.0, 0.0, -1.0, 0.5, -0.25, 2.0, 1.5, 0.75, -0.5, -1.0, 1.0, 0.25,
            ],
        );
        let bias = tensor([4], &[0.5, -0.25, 1.0, -0.75]);
        let transposed = weight.transpose(0, 1).unwrap();
        let linear = Linear::new(weight, Some(bias.clone())).unwrap();
        let input_values = (0..36)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 5.0)
            .collect::<Vec<_>>();
        let input = Tensor::new_f32(vec![2, 3, 2, 3], input_values);

        let expected = input.matmul(&transposed).unwrap().into_add(&bias).unwrap();
        let actual = linear.forward(&input).unwrap();
        assert_eq!(actual.shape(), [2, 3, 2, 4]);
        assert_tensors_close(&actual, &expected);

        let expected_softmax = expected.clone().into_softmax(-1).unwrap();
        let actual = linear.forward_softmax(&input).unwrap();
        assert_tensors_close(&actual, &expected_softmax);
    }

    fn assert_tensors_close(actual: &Tensor, expected: &Tensor) {
        assert_eq!(actual.shape(), expected.shape());
        for (&actual, &expected) in actual
            .as_f32()
            .unwrap()
            .iter()
            .zip(expected.as_f32().unwrap())
        {
            assert!((actual - expected).abs() <= 3.0e-5 * (1.0 + expected.abs()));
        }
    }

    #[test]
    fn layer_norm_normalizes_the_last_dimension() {
        let norm = LayerNorm::new(
            tensor([3], &[1.0, 1.0, 1.0]),
            tensor([3], &[0.0, 0.0, 0.0]),
            1e-5,
        )
        .unwrap();
        let output = norm
            .forward(&tensor([2, 3], &[1.0, 2.0, 3.0, 4.0, 4.0, 4.0]))
            .unwrap();
        let rows = output.as_f32().unwrap();
        assert!(rows[..3].iter().copied().sum::<f32>().abs() < 1e-5);
        assert!(rows[3..].iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn convolution_wrapper_runs_packed_pointwise_weight() {
        let convolution = Conv2d::new(
            tensor([2, 2, 1, 1], &[1.0, 2.0, 3.0, 4.0]),
            Some(tensor([2], &[0.5, -0.5])),
            [1, 1],
            [0; 4],
            1,
        )
        .unwrap();
        let output = convolution
            .forward(&tensor([1, 2, 1, 2], &[1.0, 2.0, 10.0, 20.0]))
            .unwrap();
        assert_eq!(output.shape(), [1, 2, 1, 2]);
        assert_eq!(output.as_f32().unwrap(), &[21.5, 42.5, 42.5, 85.5]);
    }

    #[test]
    fn transposed_convolution_wrapper_uses_io_weight_layout() {
        let convolution = ConvTranspose2d::new(
            tensor([1, 1, 2, 2], &[1.0; 4]),
            Some(tensor([1], &[0.5])),
            [2, 2],
            [0; 4],
            1,
        )
        .unwrap();
        let output = convolution
            .forward(&tensor([1, 1, 1, 2], &[1.0, 2.0]))
            .unwrap();
        assert_eq!(output.shape(), [1, 1, 2, 4]);
        assert_eq!(
            output.as_f32().unwrap(),
            &[1.5, 1.5, 2.5, 2.5, 1.5, 1.5, 2.5, 2.5]
        );
    }
}
