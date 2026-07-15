use super::{
    kernels,
    ops::{ConvOptions, ExactSparseConvWeights, Node, Operation, PoolOptions},
    tensor::{IntoShape, Tensor, element_count},
};
use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
#[cfg(feature = "cpu-profile")]
use std::time::Instant;

fn run(operation: Operation, inputs: Vec<Tensor>) -> Result<Tensor> {
    Node {
        name: String::new(),
        operation,
    }
    .run(inputs)
}

#[cfg(feature = "cpu-profile")]
fn profile_direct(operation: &str, output: &Tensor, started: Instant) {
    eprintln!(
        "cpu-profile operation={operation} output={:?} elapsed_ms={:.6}",
        output.shape(),
        started.elapsed().as_secs_f64() * 1_000.0
    );
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

    #[cfg(test)]
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
        // Sparse storage is lossless: only complete blocks of exact zeros are omitted.
        let exact_sparse_weights = ((tiled_spatial || sparse_pointwise)
            && output_channels.is_multiple_of(4))
        .then(|| {
            ExactSparseConvWeights::from_dense(
                weight.as_f32().expect("Conv weight was validated as F32"),
                output_channels,
                inner,
            )
        })
        .flatten();
        let system_dense_pointwise = cfg!(target_os = "macos")
            && groups == 1
            && kernel_height == 1
            && kernel_width == 1
            && strides == [1, 1]
            && pads == [0; 4];
        let system_dense_spatial = cfg!(target_os = "macos")
            && groups == 1
            && (kernel_height != 1 || kernel_width != 1)
            && exact_sparse_weights.is_none();
        let direct_spatial = groups == 1
            && (kernel_height != 1 || kernel_width != 1)
            && strides.into_iter().all(|stride| matches!(stride, 1 | 2))
            && output_channels.is_multiple_of(4)
            && exact_sparse_weights.is_none()
            && kernels::supports_direct_spatial_conv();
        // Six output rows by sixteen contiguous spatial columns keeps the
        // same accumulator count as the twelve-by-eight kernel while making
        // better use of the wide NCHW feature planes used by these models.
        let blocked_pointwise = cfg!(target_arch = "x86_64")
            && groups == 1
            && output_channels >= 128
            && inner >= 96
            && kernel_height == 1
            && kernel_width == 1
            && strides == [1, 1]
            && pads == [0; 4]
            && exact_sparse_weights.is_none();
        let packed_pointwise = groups == 1
            && ((kernel_height == 1 && kernel_width == 1)
                || (strides != [1, 1] && output_channels >= 48)
                || tiled_spatial)
            && !system_dense_pointwise
            && !system_dense_spatial
            && !direct_spatial;
        let weight = if direct_spatial {
            pack_conv_rows(weight, output_channels, 6)?
        } else if packed_pointwise {
            if tiled_spatial || exact_sparse_weights.is_some() {
                pack_conv_rows(weight, output_channels, 4)?
            } else if blocked_pointwise {
                pack_conv_rows(weight, output_channels, 6)?
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
                direct_spatial,
                packed_pointwise,
                blocked_pointwise,
                system_dense_pointwise,
                system_dense_spatial,
                exact_sparse_weights,
            },
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, None)
    }

    pub(crate) fn forward_relu(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, Some(kernels::UnaryOperation::Relu))
    }

    pub(crate) fn forward_silu(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, Some(kernels::UnaryOperation::Silu))
    }

    pub(crate) fn forward_gelu(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, Some(kernels::UnaryOperation::Gelu))
    }

    pub(crate) fn forward_pointwise_pair_gelu(
        &self,
        second: &Self,
        input: Tensor,
        residual: bool,
    ) -> Result<Tensor> {
        let first_shape: Option<[usize; 4]> = self.weight.shape.as_slice().try_into().ok();
        let second_shape: Option<[usize; 4]> = second.weight.shape.as_slice().try_into().ok();
        let first_block_rows = self.packed_pointwise_block_rows();
        let second_block_rows = second.packed_pointwise_block_rows();
        let input_shape = input.dims4().ok();
        let compatible = match (first_shape, second_shape, input_shape) {
            (
                Some([hidden_channels, input_channels, 1, 1]),
                Some([output_channels, second_input_channels, 1, 1]),
                Some((batch, actual_input_channels, height, width)),
            ) => {
                batch > 0
                    && height > 0
                    && width > 0
                    && actual_input_channels == input_channels
                    && second_input_channels == hidden_channels
                    && (!residual || output_channels == input_channels)
            }
            _ => false,
        };

        if !compatible || first_block_rows.is_none() || second_block_rows.is_none() {
            let hidden = self.forward_gelu(&input)?;
            let output = second.forward(&hidden)?;
            return if residual {
                input.into_add(&output)
            } else {
                Ok(output)
            };
        }

        #[cfg(feature = "cpu-profile")]
        let started = Instant::now();
        let [hidden_channels, input_channels, _, _] = first_shape.expect("compatible shape");
        let [output_channels, _, _, _] = second_shape.expect("compatible shape");
        let (batch, _, height, width) = input_shape.expect("compatible input");
        let plane = height
            .checked_mul(width)
            .context("pointwise pair feature plane overflow")?;
        let input_batch = input_channels
            .checked_mul(plane)
            .context("pointwise pair input batch overflow")?;
        let output_batch = output_channels
            .checked_mul(plane)
            .context("pointwise pair output batch overflow")?;
        let output_len = batch
            .checked_mul(output_batch)
            .context("pointwise pair output length overflow")?;
        let first_block_rows = first_block_rows.expect("compatible packing");
        let second_block_rows = second_block_rows.expect("compatible packing");

        const TILE_COLUMNS: usize = 16;
        // Small NCHW planes are already cache-resident and can be consumed
        // directly. Larger planes benefit from a compact tile before GEMM.
        const PACK_INPUT_MIN_ELEMENTS: usize = 128 * 1024;
        let pack_input = input_batch >= PACK_INPUT_MIN_ELEMENTS;
        let mut input_tile = if pack_input {
            vec![0.0; input_channels * TILE_COLUMNS]
        } else {
            Vec::new()
        };
        let mut hidden_tile = vec![0.0; hidden_channels * TILE_COLUMNS];
        let mut output_tile = vec![0.0; output_channels * TILE_COLUMNS];

        if residual {
            let mut output = input;
            let values = output.f32_mut()?;
            for batch_index in 0..batch {
                let batch_start = batch_index * input_batch;
                for column_start in (0..plane).step_by(TILE_COLUMNS) {
                    let columns = (plane - column_start).min(TILE_COLUMNS);
                    let (first_input, first_stride) = if pack_input {
                        for input_channel in 0..input_channels {
                            let source = batch_start + input_channel * plane + column_start;
                            let destination = input_channel * columns;
                            input_tile[destination..destination + columns]
                                .copy_from_slice(&values[source..source + columns]);
                        }
                        (&input_tile[..input_channels * columns], columns)
                    } else {
                        (&values[batch_start + column_start..], plane)
                    };
                    self.run_packed_pointwise_tile(
                        &mut hidden_tile[..hidden_channels * columns],
                        first_input,
                        input_channels,
                        columns,
                        first_stride,
                        first_block_rows,
                        Some(kernels::UnaryOperation::Gelu),
                    )?;
                    second.run_packed_pointwise_tile(
                        &mut output_tile[..output_channels * columns],
                        &hidden_tile[..hidden_channels * columns],
                        hidden_channels,
                        columns,
                        columns,
                        second_block_rows,
                        None,
                    )?;
                    for output_channel in 0..output_channels {
                        let output_start = batch_start + output_channel * plane + column_start;
                        let tile_start = output_channel * columns;
                        kernels::add_in_place(
                            &mut values[output_start..output_start + columns],
                            &output_tile[tile_start..tile_start + columns],
                        );
                    }
                }
            }
            #[cfg(feature = "cpu-profile")]
            profile_direct("PointwisePairGelu", &output, started);
            Ok(output)
        } else {
            let input_values = input.as_f32()?;
            let mut output_values = vec![0.0; output_len];
            for batch_index in 0..batch {
                let input_start = batch_index * input_batch;
                let output_start = batch_index * output_batch;
                for column_start in (0..plane).step_by(TILE_COLUMNS) {
                    let columns = (plane - column_start).min(TILE_COLUMNS);
                    let (first_input, first_stride) = if pack_input {
                        for input_channel in 0..input_channels {
                            let source = input_start + input_channel * plane + column_start;
                            let destination = input_channel * columns;
                            input_tile[destination..destination + columns]
                                .copy_from_slice(&input_values[source..source + columns]);
                        }
                        (&input_tile[..input_channels * columns], columns)
                    } else {
                        (&input_values[input_start + column_start..], plane)
                    };
                    self.run_packed_pointwise_tile(
                        &mut hidden_tile[..hidden_channels * columns],
                        first_input,
                        input_channels,
                        columns,
                        first_stride,
                        first_block_rows,
                        Some(kernels::UnaryOperation::Gelu),
                    )?;
                    second.run_packed_pointwise_tile(
                        &mut output_tile[..output_channels * columns],
                        &hidden_tile[..hidden_channels * columns],
                        hidden_channels,
                        columns,
                        columns,
                        second_block_rows,
                        None,
                    )?;
                    for output_channel in 0..output_channels {
                        let destination = output_start + output_channel * plane + column_start;
                        let tile_start = output_channel * columns;
                        output_values[destination..destination + columns]
                            .copy_from_slice(&output_tile[tile_start..tile_start + columns]);
                    }
                }
            }
            let output =
                Tensor::new_f32(vec![batch, output_channels, height, width], output_values);
            #[cfg(feature = "cpu-profile")]
            profile_direct("PointwisePairGelu", &output, started);
            Ok(output)
        }
    }

    pub(crate) fn forward_depthwise_pointwise(
        &self,
        pointwise: &Self,
        input: &Tensor,
    ) -> Result<Tensor> {
        let depthwise_shape: Option<[usize; 4]> = self.weight.shape.as_slice().try_into().ok();
        let pointwise_shape: Option<[usize; 4]> = pointwise.weight.shape.as_slice().try_into().ok();
        let pointwise_block_rows = pointwise.packed_pointwise_block_rows();
        let input_shape = input.dims4().ok();
        let compatible = match (depthwise_shape, pointwise_shape, input_shape) {
            (
                Some([channels, 1, kernel_height, kernel_width]),
                Some([_, pointwise_input_channels, 1, 1]),
                Some((batch, input_channels, height, width)),
            ) => {
                let padding = kernel_height / 2;
                batch > 0
                    && height > 0
                    && width > 0
                    && channels == input_channels
                    && pointwise_input_channels == channels
                    && kernel_height == kernel_width
                    && matches!(kernel_height, 3 | 5 | 7 | 9)
                    && self.options.groups == channels
                    && self.options.strides == [1, 1]
                    && self.options.pads == [padding; 4]
            }
            _ => false,
        };

        if !compatible || pointwise_block_rows.is_none() {
            return pointwise.forward(&self.forward(input)?);
        }

        #[cfg(feature = "cpu-profile")]
        let started = Instant::now();
        let [channels, _, kernel, _] = depthwise_shape.expect("compatible depthwise shape");
        let [output_channels, _, _, _] = pointwise_shape.expect("compatible pointwise shape");
        let (batch, _, height, width) = input_shape.expect("compatible input");
        let plane = height
            .checked_mul(width)
            .context("depthwise-pointwise feature plane overflow")?;
        let input_batch = channels
            .checked_mul(plane)
            .context("depthwise-pointwise input batch overflow")?;
        let output_batch = output_channels
            .checked_mul(plane)
            .context("depthwise-pointwise output batch overflow")?;
        let output_len = batch
            .checked_mul(output_batch)
            .context("depthwise-pointwise output length overflow")?;
        let pointwise_block_rows = pointwise_block_rows.expect("compatible pointwise packing");
        let depthwise_weight = self.weight.as_f32()?;
        let depthwise_bias = self.bias.as_ref().map(Tensor::as_f32).transpose()?;
        let input_values = input.as_f32()?;

        const STRIP_ROWS: usize = 4;
        const TILE_COLUMNS: usize = 16;
        let maximum_strip_rows = height.min(STRIP_ROWS);
        let maximum_strip_plane = maximum_strip_rows
            .checked_mul(width)
            .context("depthwise-pointwise strip plane overflow")?;
        let strip_len = channels
            .checked_mul(maximum_strip_plane)
            .context("depthwise-pointwise strip length overflow")?;
        let tile_len = output_channels
            .checked_mul(TILE_COLUMNS)
            .context("depthwise-pointwise tile length overflow")?;
        let mut depthwise_strip = vec![0.0; strip_len];
        let mut output_tile = vec![0.0; tile_len];
        let mut output_values = vec![0.0; output_len];

        for batch_index in 0..batch {
            let input_start = batch_index * input_batch;
            let output_start = batch_index * output_batch;
            let batch_input = &input_values[input_start..input_start + input_batch];
            for y_start in (0..height).step_by(STRIP_ROWS) {
                let rows = (height - y_start).min(STRIP_ROWS);
                let strip_plane = rows * width;
                let strip = &mut depthwise_strip[..channels * strip_plane];
                kernels::depthwise_conv2d_same_strip(
                    strip,
                    batch_input,
                    depthwise_weight,
                    depthwise_bias,
                    channels,
                    height,
                    width,
                    kernel,
                    y_start,
                    rows,
                );

                for column_start in (0..strip_plane).step_by(TILE_COLUMNS) {
                    let columns = (strip_plane - column_start).min(TILE_COLUMNS);
                    pointwise.run_packed_pointwise_tile(
                        &mut output_tile[..output_channels * columns],
                        &strip[column_start..],
                        channels,
                        columns,
                        strip_plane,
                        pointwise_block_rows,
                        None,
                    )?;
                    let output_column = y_start * width + column_start;
                    for output_channel in 0..output_channels {
                        let destination = output_start + output_channel * plane + output_column;
                        let tile_start = output_channel * columns;
                        output_values[destination..destination + columns]
                            .copy_from_slice(&output_tile[tile_start..tile_start + columns]);
                    }
                }
            }
        }

        let output = Tensor::new_f32(vec![batch, output_channels, height, width], output_values);
        #[cfg(feature = "cpu-profile")]
        profile_direct("DepthwisePointwise", &output, started);
        Ok(output)
    }

    fn packed_pointwise_block_rows(&self) -> Option<usize> {
        (kernels::supports_pointwise_pair_fusion()
            && self.options.groups == 1
            && self.options.strides == [1, 1]
            && self.options.pads == [0; 4]
            && self.options.packed_pointwise
            && !self.options.system_dense_pointwise
            && self.options.exact_sparse_weights.is_none())
        .then_some(if self.options.blocked_pointwise {
            6
        } else {
            12
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_packed_pointwise_tile(
        &self,
        output: &mut [f32],
        input: &[f32],
        input_channels: usize,
        columns: usize,
        input_stride: usize,
        block_rows: usize,
        activation: Option<kernels::UnaryOperation>,
    ) -> Result<()> {
        let output_channels = self.weight.shape[0];
        let bias = self.bias.as_ref().map(Tensor::as_f32).transpose()?;
        kernels::gemm_packed_left_tile(
            output,
            self.weight.as_f32()?,
            input,
            output_channels,
            input_channels,
            columns,
            input_stride,
            bias,
            activation,
            block_rows,
        );
        Ok(())
    }

    fn run(&self, input: &Tensor, activation: Option<kernels::UnaryOperation>) -> Result<Tensor> {
        let mut inputs = vec![input.clone(), self.weight.clone()];
        if let Some(bias) = &self.bias {
            inputs.push(bias.clone());
        }
        let operation = match activation {
            None => Operation::Conv(self.options.clone()),
            Some(kernels::UnaryOperation::Gelu) => Operation::ConvGelu(self.options.clone()),
            Some(kernels::UnaryOperation::Relu) => Operation::ConvRelu(self.options.clone()),
            Some(kernels::UnaryOperation::Silu) => Operation::ConvSilu(self.options.clone()),
            Some(_) => unreachable!("unsupported fused Conv activation"),
        };
        run(operation, inputs)
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
                direct_spatial: false,
                packed_pointwise: false,
                blocked_pointwise: false,
                system_dense_pointwise: false,
                system_dense_spatial: false,
                exact_sparse_weights: None,
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
            kernels::layer_norm_in_place(row, weight, bias, self.epsilon);
        });
        Ok(Tensor::new_f32(input.shape.clone(), output))
    }
}

#[derive(Clone)]
pub(crate) struct Linear {
    // x86 kernels normally consume eight-column blocks. Large classifiers use
    // sixteen-column blocks; other targets keep native [out, in] weights.
    weight: Tensor,
    input_features: usize,
    output_features: usize,
    bias: Option<Tensor>,
    #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
    weight_block_columns: usize,
}

#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
const fn use_linear_6x16(input_features: usize, output_features: usize) -> bool {
    input_features >= 192 && output_features >= 4_096
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
        #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
        let weight_block_columns = if use_linear_6x16(input_features, output_features) {
            16
        } else {
            8
        };
        #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
        let weight = pack_conv_rows(weight, output_features, weight_block_columns)?;
        Ok(Self {
            weight,
            input_features,
            output_features,
            bias,
            #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
            weight_block_columns,
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, None, false)
    }

    pub(crate) fn forward_silu(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, Some(kernels::UnaryOperation::Silu), false)
    }

    pub(crate) fn forward_softmax(&self, input: &Tensor) -> Result<Tensor> {
        self.run(input, None, true)
    }

    fn run(
        &self,
        input: &Tensor,
        activation: Option<kernels::UnaryOperation>,
        apply_softmax: bool,
    ) -> Result<Tensor> {
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

        #[cfg(feature = "cpu-profile")]
        let started = Instant::now();
        let input = input.as_f32()?;
        let weight = self.weight.as_f32()?;
        let bias = self.bias.as_ref().map(Tensor::as_f32).transpose()?;
        let mut output = vec![0.0; output_len];
        #[cfg(target_os = "macos")]
        kernels::linear_system_dense(
            &mut output,
            input,
            weight,
            rows,
            self.input_features,
            self.output_features,
            bias,
            apply_softmax,
        );
        #[cfg(target_os = "macos")]
        if let Some(activation) = activation {
            kernels::unary_in_place(&mut output, activation);
        }
        #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
        kernels::linear_right_transposed(
            &mut output,
            input,
            weight,
            rows,
            self.input_features,
            self.output_features,
            self.weight_block_columns,
            bias,
            activation,
            apply_softmax,
        );
        #[cfg(all(not(target_os = "macos"), not(target_arch = "x86_64")))]
        kernels::linear_right_transposed(
            &mut output,
            input,
            weight,
            rows,
            self.input_features,
            self.output_features,
            bias,
            activation,
            apply_softmax,
        );
        #[cfg(feature = "cpu-profile")]
        eprintln!(
            "cpu-profile operation=Linear output={output_shape:?} elapsed_ms={:.6}",
            started.elapsed().as_secs_f64() * 1_000.0
        );
        Ok(Tensor::new_f32(output_shape, output))
    }
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
    fn linear_fuses_bias_and_softmax() {
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
    fn direct_linear_matches_dynamic_matmul_with_leading_dimensions() {
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

        let empty = Tensor::new_f32(vec![2, 0, 3], Vec::new());
        let actual = linear.forward(&empty).unwrap();
        assert_eq!(actual.shape(), [2, 0, 4]);
        assert!(actual.as_f32().unwrap().is_empty());
    }

    #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
    #[test]
    fn linear_6x16_packing_is_limited_to_large_classifiers() {
        assert!(!use_linear_6x16(80, 6_906));
        assert!(!use_linear_6x16(120, 18_710));
        assert!(use_linear_6x16(192, 18_710));
        assert!(!use_linear_6x16(191, 18_710));
        assert!(!use_linear_6x16(192, 4_095));
        assert!(!use_linear_6x16(768, 192));
        assert!(use_linear_6x16(192, 4_096));
        assert!(use_linear_6x16(768, 18_710));
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
    fn pointwise_pair_tile_matches_separate_packed_12_convolutions() {
        let (batch, input_channels, hidden_channels, output_channels, height, width) =
            (2, 5, 13, 7, 2, 9);
        let values = |length: usize, multiplier: usize, divisor: f32| {
            (0..length)
                .map(|index| ((index * multiplier % 29) as f32 - 14.0) / divisor)
                .collect::<Vec<_>>()
        };
        let first = Conv2d::new(
            Tensor::new_f32(
                vec![hidden_channels, input_channels, 1, 1],
                values(hidden_channels * input_channels, 11, 19.0),
            ),
            Some(Tensor::new_f32(
                vec![hidden_channels],
                values(hidden_channels, 7, 23.0),
            )),
            [1, 1],
            [0; 4],
            1,
        )
        .unwrap();
        let second = Conv2d::new(
            Tensor::new_f32(
                vec![output_channels, hidden_channels, 1, 1],
                values(output_channels * hidden_channels, 13, 17.0),
            ),
            Some(Tensor::new_f32(
                vec![output_channels],
                values(output_channels, 5, 31.0),
            )),
            [1, 1],
            [0; 4],
            1,
        )
        .unwrap();
        let input = Tensor::new_f32(
            vec![batch, input_channels, height, width],
            values(batch * input_channels * height * width, 17, 13.0),
        );

        let expected = second
            .forward(&first.forward_gelu(&input).unwrap())
            .unwrap();
        let actual = first
            .forward_pointwise_pair_gelu(&second, input, false)
            .unwrap();
        assert_tensors_close(&actual, &expected);
    }

    #[test]
    fn pointwise_pair_tile_matches_blocked_6_residual_path() {
        let (channels, hidden_channels, width) = (128, 256, 17);
        let values = |length: usize, multiplier: usize, divisor: f32| {
            (0..length)
                .map(|index| ((index * multiplier % 37) as f32 - 18.0) / divisor)
                .collect::<Vec<_>>()
        };
        let first = Conv2d::new(
            Tensor::new_f32(
                vec![hidden_channels, channels, 1, 1],
                values(hidden_channels * channels, 19, 41.0),
            ),
            Some(Tensor::new_f32(
                vec![hidden_channels],
                values(hidden_channels, 7, 43.0),
            )),
            [1, 1],
            [0; 4],
            1,
        )
        .unwrap();
        let second = Conv2d::new(
            Tensor::new_f32(
                vec![channels, hidden_channels, 1, 1],
                values(channels * hidden_channels, 23, 47.0),
            ),
            Some(Tensor::new_f32(vec![channels], values(channels, 11, 53.0))),
            [1, 1],
            [0; 4],
            1,
        )
        .unwrap();
        let input = Tensor::new_f32(
            vec![1, channels, 1, width],
            values(channels * width, 29, 31.0),
        );

        let expected = input
            .clone()
            .into_add(
                &second
                    .forward(&first.forward_gelu(&input).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let actual = first
            .forward_pointwise_pair_gelu(&second, input, true)
            .unwrap();
        assert_tensors_close(&actual, &expected);
    }

    #[test]
    fn depthwise_pointwise_strips_match_separate_kernels() {
        let values = |length: usize, multiplier: usize, modulus: usize, divisor: f32| {
            (0..length)
                .map(|index| {
                    ((index * multiplier % modulus) as f32 - (modulus / 2) as f32) / divisor
                })
                .collect::<Vec<_>>()
        };

        for (kernel, channels, output_channels, batch, depthwise_has_bias, pointwise_has_bias) in [
            (3, 128, 128, 1, true, true),
            (5, 64, 16, 2, true, false),
            (7, 96, 24, 1, true, false),
            (9, 256, 64, 1, false, true),
        ] {
            let height = 5;
            let width = 19;
            let padding = kernel / 2;
            let depthwise = Conv2d::new(
                Tensor::new_f32(
                    vec![channels, 1, kernel, kernel],
                    values(channels * kernel * kernel, 17, 43, 29.0),
                ),
                depthwise_has_bias
                    .then(|| Tensor::new_f32(vec![channels], values(channels, 11, 37, 31.0))),
                [1, 1],
                [padding; 4],
                channels,
            )
            .unwrap();
            let pointwise = Conv2d::new(
                Tensor::new_f32(
                    vec![output_channels, channels, 1, 1],
                    values(output_channels * channels, 23, 47, 41.0),
                ),
                pointwise_has_bias.then(|| {
                    Tensor::new_f32(vec![output_channels], values(output_channels, 13, 31, 37.0))
                }),
                [1, 1],
                [0; 4],
                1,
            )
            .unwrap();
            let input = Tensor::new_f32(
                vec![batch, channels, height, width],
                values(batch * channels * height * width, 29, 53, 43.0),
            );

            let expected = pointwise
                .forward(&depthwise.forward(&input).unwrap())
                .unwrap();
            let actual = depthwise
                .forward_depthwise_pointwise(&pointwise, &input)
                .unwrap();
            assert_tensors_close(&actual, &expected);
        }
    }

    #[test]
    fn depthwise_pointwise_falls_back_for_stride_two() {
        let channels = 4;
        let output_channels = 3;
        let depthwise = Conv2d::new(
            Tensor::new_f32(
                vec![channels, 1, 3, 3],
                (0..channels * 9)
                    .map(|index| (index as f32 - 17.0) / 23.0)
                    .collect(),
            ),
            Some(Tensor::new_f32(vec![channels], vec![0.125; channels])),
            [2, 2],
            [1; 4],
            channels,
        )
        .unwrap();
        let pointwise = Conv2d::new(
            Tensor::new_f32(
                vec![output_channels, channels, 1, 1],
                (0..output_channels * channels)
                    .map(|index| (index as f32 - 5.0) / 17.0)
                    .collect(),
            ),
            None,
            [1, 1],
            [0; 4],
            1,
        )
        .unwrap();
        let input = Tensor::new_f32(
            vec![1, channels, 7, 9],
            (0..channels * 7 * 9)
                .map(|index| ((index * 7 % 41) as f32 - 20.0) / 29.0)
                .collect(),
        );

        let expected = pointwise
            .forward(&depthwise.forward(&input).unwrap())
            .unwrap();
        let actual = depthwise
            .forward_depthwise_pointwise(&pointwise, &input)
            .unwrap();
        assert_tensors_close(&actual, &expected);
    }

    #[test]
    fn direct_spatial_six_row_blocks_match_scalar_with_two_and_four_row_tails() {
        if !kernels::supports_direct_spatial_conv() {
            return;
        }

        let input_channels = 2;
        let input_height = 7;
        let input_width = 35;
        let kernel = 3;
        let strides = [2, 2];
        let pads = [1, 1, 1, 1];
        let output_height = 4;
        let output_width = 18;
        let input_values = (0..input_channels * input_height * input_width)
            .map(|index| ((index * 17 % 47) as f32 - 23.0) / 29.0)
            .collect::<Vec<_>>();

        for output_channels in [8, 16] {
            let weights = (0..output_channels * input_channels * kernel * kernel)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 31.0)
                .collect::<Vec<_>>();
            let bias = (0..output_channels)
                .map(|channel| (channel as f32 - 5.0) / 17.0)
                .collect::<Vec<_>>();
            let convolution = Conv2d::new(
                Tensor::new_f32(
                    vec![output_channels, input_channels, kernel, kernel],
                    weights.clone(),
                ),
                Some(Tensor::new_f32(vec![output_channels], bias.clone())),
                strides,
                pads,
                1,
            )
            .unwrap();
            assert!(convolution.options.direct_spatial);
            let actual = convolution
                .forward_relu(&Tensor::new_f32(
                    vec![1, input_channels, input_height, input_width],
                    input_values.clone(),
                ))
                .unwrap();

            let output_plane = output_height * output_width;
            let input_plane = input_height * input_width;
            let mut expected = vec![0.0; output_channels * output_plane];
            for output_channel in 0..output_channels {
                for output_y in 0..output_height {
                    for output_x in 0..output_width {
                        let mut sum = bias[output_channel];
                        for input_channel in 0..input_channels {
                            for kernel_y in 0..kernel {
                                let padded_y = output_y * strides[0] + kernel_y;
                                if padded_y < pads[0] || padded_y - pads[0] >= input_height {
                                    continue;
                                }
                                let input_y = padded_y - pads[0];
                                for kernel_x in 0..kernel {
                                    let padded_x = output_x * strides[1] + kernel_x;
                                    if padded_x < pads[1] || padded_x - pads[1] >= input_width {
                                        continue;
                                    }
                                    let input_x = padded_x - pads[1];
                                    let input_value = input_values[input_channel * input_plane
                                        + input_y * input_width
                                        + input_x];
                                    let weight_index = ((output_channel * input_channels
                                        + input_channel)
                                        * kernel
                                        + kernel_y)
                                        * kernel
                                        + kernel_x;
                                    sum = input_value.mul_add(weights[weight_index], sum);
                                }
                            }
                        }
                        expected
                            [output_channel * output_plane + output_y * output_width + output_x] =
                            sum.max(0.0);
                    }
                }
            }
            assert_tensors_close(
                &actual,
                &Tensor::new_f32(
                    vec![1, output_channels, output_height, output_width],
                    expected,
                ),
            );
        }
    }

    #[test]
    fn exact_sparse_pointwise_preserves_dynamic_width_tail() {
        let channels = 512;
        let width = 17;
        let mut weights = vec![0.0; channels * channels];
        let mut scales = vec![0.0; channels];
        for row in 0..channels {
            let scale = (row % 13 + 1) as f32 / 17.0;
            weights[row * channels + row] = scale;
            scales[row] = scale;
        }
        let bias = (0..channels)
            .map(|row| (row % 7) as f32 / 19.0)
            .collect::<Vec<_>>();
        let convolution = Conv2d::new(
            Tensor::new_f32(vec![channels, channels, 1, 1], weights),
            Some(Tensor::new_f32(vec![channels], bias.clone())),
            [1, 1],
            [0; 4],
            1,
        )
        .unwrap();
        let input_values = (0..channels * width)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 23.0)
            .collect::<Vec<_>>();
        let output = convolution
            .forward(&Tensor::new_f32(
                vec![1, channels, 1, width],
                input_values.clone(),
            ))
            .unwrap();

        for row in 0..channels {
            for column in 0..width {
                let expected = input_values[row * width + column].mul_add(scales[row], bias[row]);
                let actual = output.as_f32().unwrap()[row * width + column];
                assert_eq!(actual, expected, "row {row}, column {column}");
            }
        }
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
