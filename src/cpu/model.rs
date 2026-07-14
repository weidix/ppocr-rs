use super::{
    backend::{
        Conv2d as BackendConv2d, ConvTranspose2d as BackendConvTranspose2d, LayerNorm, Linear,
    },
    tensor::Tensor,
    weights::{VarBuilder, Weights},
};
use anyhow::{Context, Result, ensure};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{path::Path, str::FromStr};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelSize {
    Medium,
    Small,
    Tiny,
}

impl ModelSize {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Medium => "medium",
            Self::Small => "small",
            Self::Tiny => "tiny",
        }
    }

    pub const fn recognizer_classes(self) -> usize {
        match self {
            Self::Medium | Self::Small => 18_710,
            Self::Tiny => 6_906,
        }
    }
}

impl FromStr for ModelSize {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "medium" => Ok(Self::Medium),
            "small" => Ok(Self::Small),
            "tiny" => Ok(Self::Tiny),
            _ => Err(format!(
                "unsupported model size {value:?}; expected medium, small, or tiny"
            )),
        }
    }
}

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

fn thread_pool(options: CpuOptions) -> Result<ThreadPool> {
    ensure!(options.threads > 0, "CPU thread count must be positive");
    ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .thread_name(|index| format!("ppocr-cpu-{index}"))
        .build()
        .context("create CPU inference thread pool")
}

#[derive(Clone, Copy)]
enum Activation {
    None,
    Relu,
    Silu,
    HardSigmoid,
    HardSigmoidFive,
}

impl Activation {
    fn forward(self, input: Tensor) -> Result<Tensor> {
        match self {
            Self::None => Ok(input),
            Self::Relu => input.into_relu(),
            Self::Silu => input.into_silu(),
            Self::HardSigmoid => input.into_hard_sigmoid(1.0 / 6.0, 0.5),
            Self::HardSigmoidFive => input.into_hard_sigmoid(0.2, 0.5),
        }
    }
}

struct Conv2d {
    convolution: BackendConv2d,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        pads: [usize; 4],
        bias: bool,
        groups: usize,
    ) -> Result<Self> {
        let weight = vb.get(
            [out_channels, in_channels / groups, kernel[0], kernel[1]],
            "weight",
        )?;
        let bias = bias.then(|| vb.get(out_channels, "bias")).transpose()?;
        Ok(Self {
            convolution: BackendConv2d::new(weight, bias, stride, pads, groups)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_bn(
        vb: VarBuilder<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        pads: [usize; 4],
        bias: bool,
        groups: usize,
        norm_name: &str,
    ) -> Result<Self> {
        let conv = vb.pp("convolution");
        let weight = conv.get(
            [out_channels, in_channels / groups, kernel[0], kernel[1]],
            "weight",
        )?;
        let bias = bias.then(|| conv.get(out_channels, "bias")).transpose()?;
        let (weight, bias) = fold_batch_norm(weight, bias, vb.pp(norm_name), out_channels)?;
        Ok(Self {
            convolution: BackendConv2d::new(weight, Some(bias), stride, pads, groups)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.convolution.forward(input)
    }

    fn forward_relu(&self, input: &Tensor) -> Result<Tensor> {
        self.convolution.forward_relu(input)
    }

    fn forward_silu(&self, input: &Tensor) -> Result<Tensor> {
        self.convolution.forward_silu(input)
    }

    fn forward_gelu(&self, input: &Tensor) -> Result<Tensor> {
        self.convolution.forward_gelu(input)
    }
}

struct RankOneConv2d {
    depthwise: Conv2d,
    pointwise: Conv2d,
}

impl RankOneConv2d {
    fn load(
        vb: VarBuilder<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        Self::from_fused(
            vb.get(
                [out_channels, in_channels, kernel_size, kernel_size],
                "weight",
            )?,
            vb.get(out_channels, "bias")?,
            in_channels,
            out_channels,
            kernel_size,
        )
    }

    fn from_fused(
        weight: Tensor,
        bias: Tensor,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let (depthwise, pointwise) =
            factor_rank_one_conv_weight(&weight, in_channels, out_channels, kernel_size)?;
        let padding = kernel_size / 2;
        Ok(Self {
            depthwise: Conv2d {
                convolution: BackendConv2d::new(
                    depthwise,
                    None,
                    [1, 1],
                    [padding; 4],
                    in_channels,
                )?,
            },
            pointwise: Conv2d {
                convolution: BackendConv2d::new(pointwise, Some(bias), [1, 1], [0; 4], 1)?,
            },
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.pointwise.forward(&self.depthwise.forward(input)?)
    }
}

fn factor_rank_one_conv_weight(
    weight: &Tensor,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    const POWER_ITERATIONS: usize = 4;
    const MAX_RELATIVE_ERROR: f64 = 1.0e-6;
    const MAX_ABSOLUTE_ERROR_SCALE: f64 = 1.0e-5;

    ensure!(
        kernel_size > 0 && kernel_size % 2 == 1,
        "rank-one convolution kernel must be positive and odd"
    );
    ensure!(
        weight.shape() == [out_channels, in_channels, kernel_size, kernel_size],
        "rank-one convolution weight has invalid shape {:?}",
        weight.shape()
    );
    let source = weight.as_f32()?;
    ensure!(
        source.iter().all(|value| value.is_finite()),
        "rank-one convolution weight contains non-finite values"
    );
    let spatial = kernel_size * kernel_size;
    let mut depthwise = vec![0.0f32; in_channels * spatial];
    let mut pointwise = vec![0.0f32; out_channels * in_channels];
    let mut direction = vec![0.0f64; spatial];
    let mut scales = vec![0.0f64; out_channels];

    for input_channel in 0..in_channels {
        let mut initial_output = 0;
        let mut initial_energy = 0.0f64;
        for output_channel in 0..out_channels {
            let offset = (output_channel * in_channels + input_channel) * spatial;
            let energy = source[offset..offset + spatial]
                .iter()
                .map(|&value| f64::from(value).powi(2))
                .sum::<f64>();
            if energy > initial_energy {
                initial_output = output_channel;
                initial_energy = energy;
            }
        }
        if initial_energy == 0.0 {
            continue;
        }

        let initial_offset = (initial_output * in_channels + input_channel) * spatial;
        let initial_norm = initial_energy.sqrt();
        for index in 0..spatial {
            direction[index] = f64::from(source[initial_offset + index]) / initial_norm;
        }

        // This is power iteration on each input channel's O-by-K^2 matrix.
        for _ in 0..POWER_ITERATIONS {
            for (output_channel, scale) in scales.iter_mut().enumerate() {
                let offset = (output_channel * in_channels + input_channel) * spatial;
                *scale = source[offset..offset + spatial]
                    .iter()
                    .zip(&direction)
                    .map(|(&value, &direction)| f64::from(value) * direction)
                    .sum();
            }
            let scale_energy = scales.iter().map(|scale| scale * scale).sum::<f64>();
            ensure!(
                scale_energy.is_finite() && scale_energy > 0.0,
                "rank-one convolution factorization became degenerate"
            );
            for (index, value) in direction.iter_mut().enumerate() {
                *value = scales
                    .iter()
                    .enumerate()
                    .map(|(output_channel, &scale)| {
                        let offset =
                            (output_channel * in_channels + input_channel) * spatial + index;
                        f64::from(source[offset]) * scale
                    })
                    .sum::<f64>()
                    / scale_energy;
            }
            let norm = direction
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            ensure!(
                norm.is_finite() && norm > 0.0,
                "rank-one convolution spatial factor became degenerate"
            );
            for value in &mut direction {
                *value /= norm;
            }
        }

        for (output_channel, scale) in scales.iter_mut().enumerate() {
            let offset = (output_channel * in_channels + input_channel) * spatial;
            *scale = source[offset..offset + spatial]
                .iter()
                .zip(&direction)
                .map(|(&value, &direction)| f64::from(value) * direction)
                .sum();
            pointwise[output_channel * in_channels + input_channel] = *scale as f32;
        }
        for (target, &value) in depthwise[input_channel * spatial..(input_channel + 1) * spatial]
            .iter_mut()
            .zip(&direction)
        {
            *target = value as f32;
        }
    }

    let mut maximum_source = 0.0f64;
    let mut maximum_error = 0.0f64;
    let mut source_energy = 0.0f64;
    let mut error_energy = 0.0f64;
    for output_channel in 0..out_channels {
        for input_channel in 0..in_channels {
            let source_offset = (output_channel * in_channels + input_channel) * spatial;
            let scale = pointwise[output_channel * in_channels + input_channel];
            for index in 0..spatial {
                let expected = source[source_offset + index];
                let actual = scale * depthwise[input_channel * spatial + index];
                let error = f64::from((expected - actual).abs());
                maximum_source = maximum_source.max(f64::from(expected.abs()));
                maximum_error = maximum_error.max(error);
                source_energy += f64::from(expected).powi(2);
                error_energy += error * error;
            }
        }
    }
    let relative_error = if source_energy == 0.0 {
        0.0
    } else {
        (error_energy / source_energy).sqrt()
    };
    let maximum_error_limit = maximum_source * MAX_ABSOLUTE_ERROR_SCALE + 1.0e-12;
    ensure!(
        maximum_error <= maximum_error_limit && relative_error <= MAX_RELATIVE_ERROR,
        "fused convolution is not rank-one separable: max error {maximum_error:.6e} \
         (limit {maximum_error_limit:.6e}), relative Frobenius error {relative_error:.6e} \
         (limit {MAX_RELATIVE_ERROR:.6e})"
    );

    Ok((
        Tensor::new_f32(vec![in_channels, 1, kernel_size, kernel_size], depthwise),
        Tensor::new_f32(vec![out_channels, in_channels, 1, 1], pointwise),
    ))
}

fn fold_batch_norm(
    weight: Tensor,
    bias: Option<Tensor>,
    norm: VarBuilder<'_>,
    channels: usize,
) -> Result<(Tensor, Tensor)> {
    let shape = weight.shape().to_vec();
    ensure!(
        shape.first() == Some(&channels),
        "batch-normalized convolution has invalid weight shape {shape:?}"
    );
    let mut weight_values = weight.into_f32()?;
    let mut bias_values = match bias {
        Some(bias) => bias.into_f32()?,
        None => vec![0.0; channels],
    };
    let gamma = norm.get(channels, "weight")?;
    let beta = norm.get(channels, "bias")?;
    let mean = norm.get(channels, "running_mean")?;
    let variance = norm.get(channels, "running_var")?;
    let gamma = gamma.as_f32()?;
    let beta = beta.as_f32()?;
    let mean = mean.as_f32()?;
    let variance = variance.as_f32()?;
    let row = weight_values.len() / channels;
    for channel in 0..channels {
        let scale = gamma[channel] / (variance[channel] + 1e-5).sqrt();
        for value in &mut weight_values[channel * row..(channel + 1) * row] {
            *value *= scale;
        }
        bias_values[channel] = (bias_values[channel] - mean[channel]) * scale + beta[channel];
    }
    Ok((
        Tensor::new_f32(shape, weight_values),
        Tensor::new_f32(vec![channels], bias_values),
    ))
}

struct ConvBnAct {
    conv: Conv2d,
    activation: Activation,
}

impl ConvBnAct {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        pads: [usize; 4],
        bias: bool,
        groups: usize,
        norm_name: &str,
        activation: Activation,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::load_bn(
                vb,
                in_channels,
                out_channels,
                kernel,
                stride,
                pads,
                bias,
                groups,
                norm_name,
            )?,
            activation,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation {
            Activation::None => self.conv.forward(input),
            Activation::Relu => self.conv.forward_relu(input),
            Activation::Silu => self.conv.forward_silu(input),
            Activation::HardSigmoid | Activation::HardSigmoidFive => {
                self.activation.forward(self.conv.forward(input)?)
            }
        }
    }
}

struct SqueezeExcitation {
    reduce: Conv2d,
    expand: Conv2d,
}

impl SqueezeExcitation {
    fn load(vb: VarBuilder<'_>, channels: usize) -> Result<Self> {
        Ok(Self {
            reduce: Conv2d::load(
                vb.pp("convolutions").pp(0),
                channels,
                channels / 4,
                [1, 1],
                [1, 1],
                [0; 4],
                true,
                1,
            )?,
            expand: Conv2d::load(
                vb.pp("convolutions").pp(2),
                channels / 4,
                channels,
                [1, 1],
                [1, 1],
                [0; 4],
                true,
                1,
            )?,
        })
    }

    fn forward(&self, input: Tensor) -> Result<Tensor> {
        let pooled = input.global_avg_pool2d()?;
        let reduced = self.reduce.forward_relu(&pooled)?;
        let attention = Activation::HardSigmoid.forward(self.expand.forward(&reduced)?)?;
        input.into_mul(&attention)
    }
}

enum TokenConv {
    Direct(Conv2d),
    ConvBn(ConvBnAct),
}

impl TokenConv {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Direct(conv) => conv.forward(input),
            Self::ConvBn(conv) => conv.forward(input),
        }
    }
}

struct LcNetBlock {
    token_conv: TokenConv,
    squeeze_excitation: Option<SqueezeExcitation>,
    channel_conv1: ConvBnAct,
    channel_conv2: ConvBnAct,
    residual: bool,
}

#[derive(Clone, Copy)]
struct BlockSpec {
    kernel: usize,
    in_channels: usize,
    out_channels: usize,
    stride: [usize; 2],
    use_se: bool,
}

impl LcNetBlock {
    fn load(vb: VarBuilder<'_>, spec: BlockSpec) -> Result<Self> {
        let residual = spec.in_channels == spec.out_channels && spec.stride == [1, 1];
        let padding = spec.kernel / 2;
        let token_conv = if residual {
            TokenConv::Direct(Conv2d::load(
                vb.pp("token_conv"),
                spec.in_channels,
                spec.out_channels,
                [spec.kernel; 2],
                spec.stride,
                [padding, padding, padding, padding],
                true,
                spec.in_channels,
            )?)
        } else {
            TokenConv::ConvBn(ConvBnAct::load(
                vb.pp("token_conv"),
                spec.in_channels,
                spec.in_channels,
                [spec.kernel; 2],
                spec.stride,
                [padding, padding, padding, padding],
                false,
                spec.in_channels,
                "normalization",
                Activation::None,
            )?)
        };
        Ok(Self {
            token_conv,
            squeeze_excitation: spec
                .use_se
                .then(|| {
                    SqueezeExcitation::load(vb.pp("token_squeeze_excitation"), spec.in_channels)
                })
                .transpose()?,
            channel_conv1: ConvBnAct::load(
                vb.pp("channel_conv1"),
                spec.in_channels,
                spec.in_channels * 2,
                [1, 1],
                [1, 1],
                [0; 4],
                false,
                1,
                "normalization",
                Activation::None,
            )?,
            channel_conv2: ConvBnAct::load(
                vb.pp("channel_conv2"),
                spec.in_channels * 2,
                spec.out_channels,
                [1, 1],
                [1, 1],
                [0; 4],
                false,
                1,
                "normalization",
                Activation::None,
            )?,
            residual,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.token_conv.forward(input)?;
        let output = match &self.squeeze_excitation {
            Some(se) => se.forward(output)?,
            None => output,
        };
        let shortcut = output;
        let output = self.channel_conv1.conv.forward_gelu(&shortcut)?;
        let output = self.channel_conv2.forward(&output)?;
        if self.residual {
            shortcut.into_add(&output)
        } else {
            Ok(output)
        }
    }
}

struct LcNetStage {
    blocks: Vec<LcNetBlock>,
}

impl LcNetStage {
    fn load(vb: VarBuilder<'_>, specs: &[BlockSpec]) -> Result<Self> {
        let blocks = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| LcNetBlock::load(vb.pp("blocks").pp(index), *spec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { blocks })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.blocks
            .iter()
            .try_fold(input.clone(), |hidden, block| block.forward(&hidden))
    }
}

struct LargeStem {
    stem1: ConvBnAct,
    stem2a: ConvBnAct,
    stem2b: ConvBnAct,
    stem3: ConvBnAct,
    stem4: ConvBnAct,
}

impl LargeStem {
    fn load(
        vb: VarBuilder<'_>,
        mid_channels: usize,
        out_channels: usize,
        activation: Activation,
    ) -> Result<Self> {
        let conv = |name, in_channels, out_channels, kernel, stride, pads| {
            ConvBnAct::load(
                vb.pp(name),
                in_channels,
                out_channels,
                kernel,
                stride,
                pads,
                false,
                1,
                "normalization",
                activation,
            )
        };
        Ok(Self {
            stem1: conv("stem1", 3, mid_channels, [3, 3], [2, 2], [1; 4])?,
            stem2a: conv(
                "stem2a",
                mid_channels,
                mid_channels / 2,
                [2, 2],
                [1, 1],
                [0, 0, 1, 1],
            )?,
            stem2b: conv(
                "stem2b",
                mid_channels / 2,
                mid_channels,
                [2, 2],
                [1, 1],
                [0, 0, 1, 1],
            )?,
            stem3: conv(
                "stem3",
                mid_channels * 2,
                mid_channels,
                [3, 3],
                [2, 2],
                [1; 4],
            )?,
            stem4: conv("stem4", mid_channels, out_channels, [1, 1], [1, 1], [0; 4])?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let embedding = self.stem1.forward(input)?;
        let branch = self.stem2a.forward(&embedding)?;
        let branch = self.stem2b.forward(&branch)?;
        let pooled = embedding.max_pool2d([2, 2], [1, 1], [0, 0, 1, 1], false)?;
        let merged = Tensor::cat(&[&pooled, &branch], 1)?;
        self.stem4.forward(&self.stem3.forward(&merged)?)
    }
}

struct SmallStem {
    conv1: ConvBnAct,
    conv2: ConvBnAct,
}

impl SmallStem {
    fn load(vb: VarBuilder<'_>, mid_channels: usize, out_channels: usize) -> Result<Self> {
        Ok(Self {
            conv1: ConvBnAct::load(
                vb.pp("conv1"),
                3,
                mid_channels,
                [3, 3],
                [2, 2],
                [1; 4],
                false,
                1,
                "normalization",
                Activation::None,
            )?,
            conv2: ConvBnAct::load(
                vb.pp("conv2"),
                mid_channels,
                out_channels,
                [3, 3],
                [2, 2],
                [1; 4],
                false,
                1,
                "normalization",
                Activation::None,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.conv2.forward(&self.conv1.conv.forward_gelu(input)?)
    }
}

#[derive(Clone, Copy)]
enum StemSpec {
    Large {
        mid_channels: usize,
        out_channels: usize,
    },
    Small {
        mid_channels: usize,
        out_channels: usize,
    },
}

enum LcNetStem {
    Large(Box<LargeStem>),
    Small(Box<SmallStem>),
}

impl LcNetStem {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Large(stem) => stem.forward(input),
            Self::Small(stem) => stem.forward(input),
        }
    }
}

struct LcNetBackbone {
    stem: LcNetStem,
    stages: Vec<LcNetStage>,
}

impl LcNetBackbone {
    fn load(
        vb: VarBuilder<'_>,
        specs: &[Vec<BlockSpec>],
        stem_spec: StemSpec,
        activation: Activation,
    ) -> Result<Self> {
        let stem = match stem_spec {
            StemSpec::Large {
                mid_channels,
                out_channels,
            } => LcNetStem::Large(Box::new(LargeStem::load(
                vb.pp("convolution"),
                mid_channels,
                out_channels,
                activation,
            )?)),
            StemSpec::Small {
                mid_channels,
                out_channels,
            } => LcNetStem::Small(Box::new(SmallStem::load(
                vb.pp("convolution"),
                mid_channels,
                out_channels,
            )?)),
        };
        let stages = specs
            .iter()
            .enumerate()
            .map(|(index, stage)| LcNetStage::load(vb.pp("blocks").pp(index), stage))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { stem, stages })
    }

    fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        let mut hidden = self.stem.forward(input)?;
        let mut outputs = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            hidden = stage.forward(&hidden)?;
            outputs.push(hidden.clone());
        }
        Ok(outputs)
    }
}

struct IntraclassBlock {
    reduce: Conv2d,
    long: Conv2d,
    mid: Conv2d,
    short: Conv2d,
    final_conv: ConvBnAct,
}

fn fuse_intraclass_conv(
    symmetric_weight: Tensor,
    symmetric_bias: Tensor,
    vertical_weight: Tensor,
    vertical_bias: Tensor,
    horizontal_weight: Tensor,
    horizontal_bias: Tensor,
) -> Result<(Tensor, Tensor)> {
    let shape = symmetric_weight.shape().to_vec();
    let [out_channels, in_channels, kernel_height, kernel_width]: [usize; 4] = shape
        .as_slice()
        .try_into()
        .with_context(|| format!("expected rank-four symmetric weight, found {shape:?}"))?;
    ensure!(
        kernel_height == kernel_width && kernel_height % 2 == 1,
        "intraclass symmetric kernel must be odd and square, found {kernel_height}x{kernel_width}"
    );
    ensure!(
        vertical_weight.shape() == [out_channels, in_channels, kernel_height, 1],
        "intraclass vertical weight has invalid shape {:?}",
        vertical_weight.shape()
    );
    ensure!(
        horizontal_weight.shape() == [out_channels, in_channels, 1, kernel_width],
        "intraclass horizontal weight has invalid shape {:?}",
        horizontal_weight.shape()
    );
    for (name, bias) in [
        ("symmetric", &symmetric_bias),
        ("vertical", &vertical_bias),
        ("horizontal", &horizontal_bias),
    ] {
        ensure!(
            bias.shape() == [out_channels],
            "intraclass {name} bias has invalid shape {:?}",
            bias.shape()
        );
    }

    let vertical = vertical_weight.as_f32()?;
    let horizontal = horizontal_weight.as_f32()?;
    let mut weight = symmetric_weight.into_f32()?;
    let center = kernel_height / 2;
    for output in 0..out_channels {
        for input in 0..in_channels {
            let branch_offset = (output * in_channels + input) * kernel_height;
            let symmetric_offset = branch_offset * kernel_width;
            for row in 0..kernel_height {
                weight[symmetric_offset + row * kernel_width + center] +=
                    vertical[branch_offset + row];
            }
            for column in 0..kernel_width {
                weight[symmetric_offset + center * kernel_width + column] +=
                    horizontal[branch_offset + column];
            }
        }
    }

    let vertical_bias = vertical_bias.as_f32()?;
    let horizontal_bias = horizontal_bias.as_f32()?;
    let mut bias = symmetric_bias.into_f32()?;
    for channel in 0..out_channels {
        bias[channel] += vertical_bias[channel];
        bias[channel] += horizontal_bias[channel];
    }
    Ok((
        Tensor::new_f32(shape, weight),
        Tensor::new_f32(vec![out_channels], bias),
    ))
}

fn load_fused_intraclass_conv(vb: VarBuilder<'_>, ratio: &str, kernel: usize) -> Result<Conv2d> {
    let symmetric = vb.pp(format!("symmetric_conv_long_{ratio}"));
    let vertical = vb.pp(format!("vertical_long_to_small_conv_{ratio}"));
    let horizontal = vb.pp(format!("horizontal_small_to_long_conv_{ratio}"));
    let (weight, bias) = fuse_intraclass_conv(
        symmetric.get([32, 32, kernel, kernel], "weight")?,
        symmetric.get(32, "bias")?,
        vertical.get([32, 32, kernel, 1], "weight")?,
        vertical.get(32, "bias")?,
        horizontal.get([32, 32, 1, kernel], "weight")?,
        horizontal.get(32, "bias")?,
    )?;
    let padding = kernel / 2;
    Ok(Conv2d {
        convolution: BackendConv2d::new(weight, Some(bias), [1, 1], [padding; 4], 1)?,
    })
}

impl IntraclassBlock {
    fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            reduce: Conv2d::load(
                vb.pp("conv_reduce_channel"),
                64,
                32,
                [1, 1],
                [1, 1],
                [0; 4],
                true,
                1,
            )?,
            long: load_fused_intraclass_conv(vb.clone(), "longratio", 7)?,
            mid: load_fused_intraclass_conv(vb.clone(), "midratio", 5)?,
            short: load_fused_intraclass_conv(vb.clone(), "shortratio", 3)?,
            final_conv: ConvBnAct::load(
                vb.pp("conv_final"),
                32,
                64,
                [1, 1],
                [1, 1],
                [0; 4],
                true,
                1,
                "norm",
                Activation::Relu,
            )?,
        })
    }

    fn forward(&self, input: Tensor) -> Result<Tensor> {
        let reduced = self.reduce.forward(&input)?;
        let layer7 = self.long.forward(&reduced)?;
        let layer5 = self.mid.forward(&layer7)?;
        let layer3 = self.short.forward(&layer5)?;
        input.into_add(&self.final_conv.forward(&layer3)?)
    }
}

struct DetectorNeck {
    adjust: Vec<Conv2d>,
    project: Vec<RankOneConv2d>,
    pan_head: Vec<Conv2d>,
    pan_lateral: Vec<RankOneConv2d>,
    intraclass: Vec<IntraclassBlock>,
}

impl DetectorNeck {
    fn load(vb: VarBuilder<'_>) -> Result<Self> {
        let adjust = [128, 256, 512, 896]
            .iter()
            .enumerate()
            .map(|(index, channels)| {
                Conv2d::load(
                    vb.pp("input_channel_adjustment_convolution").pp(index),
                    *channels,
                    256,
                    [1, 1],
                    [1, 1],
                    [0; 4],
                    false,
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let project = (0..4)
            .map(|index| {
                RankOneConv2d::load(
                    vb.pp("input_feature_projection_convolution").pp(index),
                    256,
                    64,
                    9,
                )
                .with_context(|| format!("factor medium neck projection {index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let pan_head = (0..3)
            .map(|index| {
                Conv2d::load(
                    vb.pp("path_aggregation_head_convolution").pp(index),
                    64,
                    64,
                    [3, 3],
                    [2, 2],
                    [1; 4],
                    false,
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let pan_lateral = (0..4)
            .map(|index| {
                RankOneConv2d::load(
                    vb.pp("path_aggregation_lateral_convolution").pp(index),
                    64,
                    64,
                    9,
                )
                .with_context(|| format!("factor medium neck lateral {index}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let intraclass = (0..4)
            .map(|index| IntraclassBlock::load(vb.pp("intraclass_blocks").pp(index)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            adjust,
            project,
            pan_head,
            pan_lateral,
            intraclass,
        })
    }

    fn forward(&self, stages: &[Tensor]) -> Result<Tensor> {
        let adjusted = self
            .adjust
            .iter()
            .zip(stages)
            .map(|(conv, feature)| conv.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        let mut top_down = Vec::with_capacity(4);
        for feature in adjusted.into_iter().rev() {
            let feature = match top_down.last() {
                Some(upper) => feature.into_add(&upsample(upper, 2)?)?,
                None => feature,
            };
            top_down.push(feature);
        }
        top_down.reverse();
        let projected = self
            .project
            .iter()
            .zip(&top_down)
            .map(|(conv, feature)| conv.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        let mut bottom_up = Vec::with_capacity(4);
        for (index, projection) in projected.into_iter().enumerate() {
            let feature = match bottom_up.last() {
                Some(lower) => projection.into_add(&self.pan_head[index - 1].forward(lower)?)?,
                None => projection,
            };
            bottom_up.push(feature);
        }
        let lateral = self
            .pan_lateral
            .iter()
            .zip(&bottom_up)
            .map(|(conv, feature)| conv.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        let refined = self
            .intraclass
            .iter()
            .zip(lateral)
            .map(|(block, feature)| block.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        let mut features = refined
            .iter()
            .zip([1, 2, 4, 8])
            .map(|(feature, scale)| upsample(feature, scale))
            .collect::<Result<Vec<_>>>()?;
        features.reverse();
        Tensor::cat(&features.iter().collect::<Vec<_>>(), 1)
    }
}

struct VariantSqueezeExcitation {
    reduce: Conv2d,
    expand: Conv2d,
}

impl VariantSqueezeExcitation {
    fn load(vb: VarBuilder<'_>, channels: usize, reduction: usize) -> Result<Self> {
        Ok(Self {
            reduce: Conv2d::load(
                vb.pp("conv1"),
                channels,
                channels / reduction,
                [1, 1],
                [1, 1],
                [0; 4],
                true,
                1,
            )?,
            expand: Conv2d::load(
                vb.pp("conv2"),
                channels / reduction,
                channels,
                [1, 1],
                [1, 1],
                [0; 4],
                true,
                1,
            )?,
        })
    }

    fn attention(&self, input: &Tensor) -> Result<Tensor> {
        let pooled = input.global_avg_pool2d()?;
        let hidden = self.reduce.forward_relu(&pooled)?;
        Activation::HardSigmoidFive.forward(self.expand.forward(&hidden)?)
    }
}

struct ResidualSqueezeExcitation {
    input: Conv2d,
    squeeze_excitation: VariantSqueezeExcitation,
}

impl ResidualSqueezeExcitation {
    fn load(
        vb: VarBuilder<'_>,
        input_channels: usize,
        output_channels: usize,
        reduction: usize,
    ) -> Result<Self> {
        Ok(Self {
            input: Conv2d::load(
                vb.pp("in_conv"),
                input_channels,
                output_channels,
                [1, 1],
                [1, 1],
                [0; 4],
                false,
                1,
            )?,
            squeeze_excitation: VariantSqueezeExcitation::load(
                vb.pp("squeeze_excitation_block"),
                output_channels,
                reduction,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.input.forward(input)?;
        let attention = self.squeeze_excitation.attention(&hidden)?;
        hidden.into_residual_mul(&attention)
    }
}

struct DepthwiseSeparableConv {
    depthwise: Conv2d,
    pointwise: Conv2d,
    squeeze_excitation: VariantSqueezeExcitation,
}

impl DepthwiseSeparableConv {
    fn load(
        vb: VarBuilder<'_>,
        channels: usize,
        kernel_size: usize,
        reduction: usize,
    ) -> Result<Self> {
        let padding = kernel_size / 2;
        Ok(Self {
            depthwise: Conv2d::load(
                vb.pp("depthwise_convolution"),
                channels,
                channels,
                [kernel_size; 2],
                [1, 1],
                [padding; 4],
                true,
                channels,
            )?,
            pointwise: Conv2d::load(
                vb.pp("pointwise_convolution"),
                channels,
                channels / 4,
                [1, 1],
                [1, 1],
                [0; 4],
                false,
                1,
            )?,
            squeeze_excitation: VariantSqueezeExcitation::load(
                vb.pp("squeeze_excitation_module"),
                channels / 4,
                reduction,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.pointwise.forward(&self.depthwise.forward(input)?)?;
        let attention = self.squeeze_excitation.attention(&hidden)?;
        hidden.into_residual_mul(&attention)
    }
}

struct RepLkFpn {
    insert: Vec<ResidualSqueezeExcitation>,
    input: Vec<DepthwiseSeparableConv>,
}

impl RepLkFpn {
    fn load(
        vb: VarBuilder<'_>,
        stage_channels: [usize; 4],
        neck_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        let insert = stage_channels
            .iter()
            .enumerate()
            .map(|(index, channels)| {
                ResidualSqueezeExcitation::load(
                    vb.pp("insert_conv").pp(index),
                    *channels,
                    neck_channels,
                    4,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let input = (0..4)
            .map(|index| {
                DepthwiseSeparableConv::load(
                    vb.pp("input_conv").pp(index),
                    neck_channels,
                    kernel_size,
                    4,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { insert, input })
    }

    fn forward(&self, stages: &[Tensor]) -> Result<Tensor> {
        let mut fused = self
            .insert
            .iter()
            .zip(stages)
            .map(|(conv, feature)| conv.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        for index in (0..3).rev() {
            let upper = upsample(&fused[index + 1], 2)?;
            let current = fused.remove(index);
            fused.insert(index, current.into_add(&upper)?);
        }
        let mut features = self
            .input
            .iter()
            .zip(&fused)
            .map(|(conv, feature)| conv.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        for (feature, scale) in features.iter_mut().zip([1, 2, 4, 8]) {
            *feature = upsample(feature, scale)?;
        }
        features.reverse();
        Tensor::cat(&features.iter().collect::<Vec<_>>(), 1)
    }
}

fn upsample(input: &Tensor, scale: usize) -> Result<Tensor> {
    if scale == 1 {
        return Ok(input.clone());
    }
    let (_, _, height, width) = input.dims4()?;
    input.resize_nearest2d([height * scale, width * scale])
}

struct TransposeConvBnRelu {
    convolution: BackendConvTranspose2d,
}

impl TransposeConvBnRelu {
    fn load(vb: VarBuilder<'_>, in_channels: usize, out_channels: usize) -> Result<Self> {
        let conv = vb.pp("convolution");
        let weight = conv.get([in_channels, out_channels, 2, 2], "weight")?;
        let bias = Some(conv.get(out_channels, "bias")?);
        let (weight, bias) =
            fold_transpose_batch_norm(weight, bias, vb.pp("norm"), in_channels, out_channels)?;
        Ok(Self {
            convolution: BackendConvTranspose2d::new(weight, Some(bias), [2, 2], [0; 4], 1)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.convolution.forward(input)?.into_relu()
    }
}

fn fold_transpose_batch_norm(
    weight: Tensor,
    bias: Option<Tensor>,
    norm: VarBuilder<'_>,
    in_channels: usize,
    out_channels: usize,
) -> Result<(Tensor, Tensor)> {
    let shape = weight.shape().to_vec();
    let mut weight_values = weight.into_f32()?;
    let mut bias_values = match bias {
        Some(bias) => bias.into_f32()?,
        None => vec![0.0; out_channels],
    };
    let gamma = norm.get(out_channels, "weight")?;
    let beta = norm.get(out_channels, "bias")?;
    let mean = norm.get(out_channels, "running_mean")?;
    let variance = norm.get(out_channels, "running_var")?;
    let gamma = gamma.as_f32()?;
    let beta = beta.as_f32()?;
    let mean = mean.as_f32()?;
    let variance = variance.as_f32()?;
    let kernel = weight_values.len() / (in_channels * out_channels);
    for out_channel in 0..out_channels {
        let scale = gamma[out_channel] / (variance[out_channel] + 1e-5).sqrt();
        for in_channel in 0..in_channels {
            let start = (in_channel * out_channels + out_channel) * kernel;
            for value in &mut weight_values[start..start + kernel] {
                *value *= scale;
            }
        }
        bias_values[out_channel] =
            (bias_values[out_channel] - mean[out_channel]) * scale + beta[out_channel];
    }
    Ok((
        Tensor::new_f32(shape, weight_values),
        Tensor::new_f32(vec![out_channels], bias_values),
    ))
}

struct DetectorHead {
    down: ConvBnAct,
    up: TransposeConvBnRelu,
    final_conv: BackendConvTranspose2d,
}

impl DetectorHead {
    fn load(vb: VarBuilder<'_>, input_channels: usize) -> Result<Self> {
        let hidden_channels = input_channels / 4;
        let down = ConvBnAct::load(
            vb.pp("conv_down"),
            input_channels,
            hidden_channels,
            [3, 3],
            [1, 1],
            [1; 4],
            false,
            1,
            "norm",
            Activation::Relu,
        )?;
        let up = TransposeConvBnRelu::load(vb.pp("conv_up"), hidden_channels, hidden_channels)?;
        let final_vb = vb.pp("conv_final");
        let final_conv = BackendConvTranspose2d::new(
            final_vb.get([hidden_channels, 1, 2, 2], "weight")?,
            Some(final_vb.get(1, "bias")?),
            [2, 2],
            [0; 4],
            1,
        )?;
        Ok(Self {
            down,
            up,
            final_conv,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.final_conv
            .forward(&self.up.forward(&self.down.forward(input)?)?)?
            .into_sigmoid()
    }
}

enum DetectorNeckKind {
    Medium(DetectorNeck),
    RepLkFpn(RepLkFpn),
}

impl DetectorNeckKind {
    fn forward(&self, stages: &[Tensor]) -> Result<Tensor> {
        match self {
            Self::Medium(neck) => neck.forward(stages),
            Self::RepLkFpn(neck) => neck.forward(stages),
        }
    }
}

pub struct Detector {
    backbone: LcNetBackbone,
    neck: DetectorNeckKind,
    head: DetectorHead,
    pool: ThreadPool,
}

impl Detector {
    pub fn load(path: impl AsRef<Path>, size: ModelSize, options: CpuOptions) -> Result<Self> {
        let pool = thread_pool(options)?;
        let weights = Weights::load(path)?;
        let vb = weights.builder();
        let encoder = vb.pp("model").pp("backbone").pp("encoder");
        let (backbone, neck, head) = match size {
            ModelSize::Medium => (
                LcNetBackbone::load(
                    encoder,
                    &detector_stages_for_channels([128, 256, 512, 896]),
                    StemSpec::Large {
                        mid_channels: 64,
                        out_channels: 128,
                    },
                    Activation::Relu,
                )?,
                DetectorNeckKind::Medium(DetectorNeck::load(vb.pp("model").pp("neck"))?),
                DetectorHead::load(vb.pp("head"), 256)?,
            ),
            ModelSize::Small => (
                LcNetBackbone::load(
                    encoder,
                    &detector_stages_for_channels([48, 96, 192, 384]),
                    StemSpec::Large {
                        mid_channels: 24,
                        out_channels: 48,
                    },
                    Activation::Relu,
                )?,
                DetectorNeckKind::RepLkFpn(RepLkFpn::load(
                    vb.pp("model").pp("neck"),
                    [48, 96, 192, 384],
                    96,
                    7,
                )?),
                DetectorHead::load(vb.pp("head"), 96)?,
            ),
            ModelSize::Tiny => (
                LcNetBackbone::load(
                    encoder,
                    &detector_stages_for_channels([32, 48, 64, 160]),
                    StemSpec::Large {
                        mid_channels: 16,
                        out_channels: 32,
                    },
                    Activation::Relu,
                )?,
                DetectorNeckKind::RepLkFpn(RepLkFpn::load(
                    vb.pp("model").pp("neck"),
                    [32, 48, 64, 160],
                    64,
                    5,
                )?),
                DetectorHead::load(vb.pp("head"), 64)?,
            ),
        };
        Ok(Self {
            backbone,
            neck,
            head,
            pool,
        })
    }

    pub fn run(&self, input: Tensor) -> Result<Tensor> {
        validate_input(&input, true)?;
        self.pool.install(|| self.forward(&input))
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;
        self.head.forward(&self.neck.forward(&features)?)
    }
}

fn validate_input(input: &Tensor, detector: bool) -> Result<()> {
    input.as_f32()?;
    let (batch, channels, height, width) = input.dims4()?;
    ensure!(batch > 0, "input batch must not be empty");
    ensure!(channels == 3, "input must have three NCHW channels");
    ensure!(
        height > 0 && width > 0,
        "input spatial dimensions must be positive"
    );
    if detector {
        ensure!(
            height % 32 == 0 && width % 32 == 0,
            "detector input height and width must be multiples of 32"
        );
    } else {
        ensure!(height == 48, "recognizer input height must be 48");
        ensure!(width >= 5, "recognizer input width must be at least 5");
    }
    Ok(())
}

fn detector_stages_for_channels(channels: [usize; 4]) -> Vec<Vec<BlockSpec>> {
    let [stage1, stage2, stage3, stage4] = channels;
    vec![
        vec![
            rec_block(stage1, stage1, [1, 1], true),
            rec_block(stage1, stage1, [1, 1], false),
        ],
        vec![
            rec_block(stage1, stage2, [2, 2], false),
            rec_block(stage2, stage2, [1, 1], true),
            rec_block(stage2, stage2, [1, 1], false),
        ],
        vec![
            rec_block(stage2, stage3, [2, 2], false),
            rec_block(stage3, stage3, [1, 1], true),
            rec_block(stage3, stage3, [1, 1], false),
            rec_block(stage3, stage3, [1, 1], true),
            rec_block(stage3, stage3, [1, 1], false),
        ],
        vec![
            rec_block(stage3, stage4, [2, 2], false),
            rec_block(stage4, stage4, [1, 1], true),
            rec_block(stage4, stage4, [1, 1], false),
        ],
    ]
}

fn load_linear(vb: VarBuilder<'_>, input: usize, output: usize) -> Result<Linear> {
    Linear::new(
        vb.get([output, input], "weight")?,
        Some(vb.get(output, "bias")?),
    )
}

fn load_layer_norm(vb: VarBuilder<'_>, features: usize) -> Result<LayerNorm> {
    LayerNorm::new(vb.get(features, "weight")?, vb.get(features, "bias")?, 1e-6)
}

struct RecAttention {
    qkv: Linear,
    projection: Linear,
    hidden_size: usize,
    num_heads: usize,
}

impl RecAttention {
    fn load(vb: VarBuilder<'_>, hidden_size: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            qkv: load_linear(vb.pp("qkv"), hidden_size, hidden_size * 3)?,
            projection: load_linear(vb.pp("projection"), hidden_size, hidden_size)?,
            hidden_size,
            num_heads,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, sequence, _) = input.dims3()?;
        let head_dim = self.hidden_size / self.num_heads;
        let qkv = self
            .qkv
            .forward(input)?
            .reshape([batch, sequence, 3, self.num_heads, head_dim])?
            .permute([2, 0, 3, 1, 4])?;
        let parts = qkv.chunk(3, 0)?;
        let query = parts[0].squeeze(0)?;
        let key = parts[1].squeeze(0)?;
        let value = parts[2].squeeze(0)?;
        let weights = query
            .matmul(&key.transpose(2, 3)?)?
            .into_affine((head_dim as f32).powf(-0.5), 0.0)?
            .into_softmax(-1)?;
        let output = weights.matmul(&value)?.transpose(1, 2)?.reshape([
            batch,
            sequence,
            self.hidden_size,
        ])?;
        self.projection.forward(&output)
    }
}

struct RecMlp {
    fc1: Linear,
    fc2: Linear,
}

impl RecMlp {
    fn load(vb: VarBuilder<'_>, hidden_size: usize, mlp_size: usize) -> Result<Self> {
        Ok(Self {
            fc1: load_linear(vb.pp("fc1"), hidden_size, mlp_size)?,
            fc2: load_linear(vb.pp("fc2"), mlp_size, hidden_size)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward_silu(input)?)
    }
}

struct RecBlock {
    layer_norm1: LayerNorm,
    attention: RecAttention,
    layer_norm2: LayerNorm,
    mlp: RecMlp,
}

impl RecBlock {
    fn load(
        vb: VarBuilder<'_>,
        hidden_size: usize,
        num_heads: usize,
        mlp_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            layer_norm1: load_layer_norm(vb.pp("layer_norm1"), hidden_size)?,
            attention: RecAttention::load(vb.pp("self_attn"), hidden_size, num_heads)?,
            layer_norm2: load_layer_norm(vb.pp("layer_norm2"), hidden_size)?,
            mlp: RecMlp::load(vb.pp("mlp"), hidden_size, mlp_size)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = input.add(&self.attention.forward(&self.layer_norm1.forward(input)?)?)?;
        let residual = self.mlp.forward(&self.layer_norm2.forward(&output)?)?;
        output.into_add(&residual)
    }
}

struct RecEncoder {
    skip: ConvBnAct,
    reduce: ConvBnAct,
    local: ConvBnAct,
    blocks: Vec<RecBlock>,
    norm: LayerNorm,
}

impl RecEncoder {
    fn load(
        vb: VarBuilder<'_>,
        input_channels: usize,
        hidden_size: usize,
        num_heads: usize,
        mlp_size: usize,
        depth: usize,
    ) -> Result<Self> {
        let load_conv = |index, in_channels, out_channels, kernel: [usize; 2], groups| {
            let padding = [kernel[0] / 2, kernel[1] / 2];
            ConvBnAct::load(
                vb.pp("conv_block").pp(index),
                in_channels,
                out_channels,
                kernel,
                [1, 1],
                [padding[0], padding[1], padding[0], padding[1]],
                false,
                groups,
                "normalization",
                Activation::Silu,
            )
        };
        Ok(Self {
            skip: load_conv(0, input_channels, hidden_size, [1, 1], 1)?,
            reduce: load_conv(1, input_channels, hidden_size, [1, 1], 1)?,
            local: load_conv(2, hidden_size, hidden_size, [1, 7], hidden_size)?,
            blocks: (0..depth)
                .map(|index| {
                    RecBlock::load(
                        vb.pp("svtr_block").pp(index),
                        hidden_size,
                        num_heads,
                        mlp_size,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            norm: load_layer_norm(vb.pp("norm"), hidden_size)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let residual = self.skip.forward(input)?;
        let hidden = self.reduce.forward(input)?;
        let local = self.local.forward(&hidden)?;
        let hidden = hidden.into_add(&local)?;
        let (batch, channels, height, width) = hidden.dims4()?;
        let mut hidden = hidden.flatten(2, 3)?.transpose(1, 2)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = self
            .norm
            .forward(&hidden)?
            .reshape([batch, height, width, channels])?
            .permute([0, 3, 1, 2])?
            .into_add(&residual)?;
        hidden.squeeze(2)?.transpose(1, 2)
    }
}

struct LightSvtrRecognizerHead {
    encoder: RecEncoder,
    classifier: Linear,
}

impl LightSvtrRecognizerHead {
    fn load(
        vb: VarBuilder<'_>,
        input_channels: usize,
        hidden_size: usize,
        mlp_size: usize,
        classes: usize,
    ) -> Result<Self> {
        Ok(Self {
            encoder: RecEncoder::load(
                vb.pp("encoder"),
                input_channels,
                hidden_size,
                8,
                mlp_size,
                2,
            )?,
            classifier: load_linear(vb.pp("head"), hidden_size, classes)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.classifier
            .forward_softmax(&self.encoder.forward(input)?)
    }
}

struct Conv1dBn {
    convolution: BackendConv2d,
}

impl Conv1dBn {
    fn load(
        conv: VarBuilder<'_>,
        norm: VarBuilder<'_>,
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = conv.get(
            [output_channels, input_channels / groups, kernel_size],
            "weight",
        )?;
        let (weight, bias) = fold_batch_norm(weight, None, norm, output_channels)?;
        let weight = weight.reshape([output_channels, input_channels / groups, 1, kernel_size])?;
        let padding = kernel_size / 2;
        Ok(Self {
            convolution: BackendConv2d::new(
                weight,
                Some(bias),
                [1, 1],
                [0, padding, 0, padding],
                groups,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.convolution.forward(input)
    }
}

struct TinyRecognizerHead {
    conv1: Conv1dBn,
    conv2: Conv1dBn,
    fc1: Linear,
    fc2: Linear,
}

impl TinyRecognizerHead {
    fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            conv1: Conv1dBn::load(vb.pp("conv1"), vb.pp("norm1"), 160, 160, 5, 160)?,
            conv2: Conv1dBn::load(vb.pp("conv2"), vb.pp("norm2"), 160, 160, 1, 1)?,
            fc1: load_linear(vb.pp("fc1"), 160, 80)?,
            fc2: load_linear(vb.pp("fc2"), 80, 6_906)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.conv1.forward(input)?.into_hard_swish()?;
        let hidden = self.conv2.forward(&hidden)?.into_hard_swish()?;
        let hidden = hidden.squeeze(2)?.transpose(1, 2)?;
        self.fc2.forward_softmax(&self.fc1.forward(&hidden)?)
    }
}

enum RecognizerHeadKind {
    LightSvtr(LightSvtrRecognizerHead),
    Tiny(TinyRecognizerHead),
}

impl RecognizerHeadKind {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::LightSvtr(head) => head.forward(input),
            Self::Tiny(head) => head.forward(input),
        }
    }
}

pub struct Recognizer {
    backbone: LcNetBackbone,
    head: RecognizerHeadKind,
    pool: ThreadPool,
}

impl Recognizer {
    pub fn load(path: impl AsRef<Path>, size: ModelSize, options: CpuOptions) -> Result<Self> {
        let pool = thread_pool(options)?;
        let weights = Weights::load(path)?;
        let vb = weights.builder();
        let encoder = vb.pp("model").pp("backbone").pp("encoder");
        let (backbone, head) = match size {
            ModelSize::Medium => (
                LcNetBackbone::load(
                    encoder,
                    &recognizer_stages(),
                    StemSpec::Large {
                        mid_channels: 64,
                        out_channels: 128,
                    },
                    Activation::Relu,
                )?,
                RecognizerHeadKind::LightSvtr(LightSvtrRecognizerHead::load(
                    vb.pp("head"),
                    768,
                    192,
                    768,
                    18_710,
                )?),
            ),
            ModelSize::Small => (
                LcNetBackbone::load(
                    encoder,
                    &small_recognizer_stages(),
                    StemSpec::Large {
                        mid_channels: 48,
                        out_channels: 96,
                    },
                    Activation::Relu,
                )?,
                RecognizerHeadKind::LightSvtr(LightSvtrRecognizerHead::load(
                    vb.pp("head"),
                    384,
                    120,
                    240,
                    18_710,
                )?),
            ),
            ModelSize::Tiny => (
                LcNetBackbone::load(
                    encoder,
                    &tiny_recognizer_stages(),
                    StemSpec::Small {
                        mid_channels: 24,
                        out_channels: 48,
                    },
                    Activation::Relu,
                )?,
                RecognizerHeadKind::Tiny(TinyRecognizerHead::load(vb.pp("head"))?),
            ),
        };
        Ok(Self {
            backbone,
            head,
            pool,
        })
    }

    pub fn run(&self, input: Tensor) -> Result<Tensor> {
        validate_input(&input, false)?;
        self.pool.install(|| self.forward(&input))
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;
        let feature = features.last().expect("recognizer backbone has stages");
        let pooled = feature.avg_pool2d([3, 2], [3, 2], [0; 4], false, false)?;
        self.head.forward(&pooled)
    }
}

fn rec_block(
    in_channels: usize,
    out_channels: usize,
    stride: [usize; 2],
    use_se: bool,
) -> BlockSpec {
    BlockSpec {
        kernel: 3,
        in_channels,
        out_channels,
        stride,
        use_se,
    }
}

fn recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![rec_block(128, 128, [1, 1], true)],
        vec![
            rec_block(128, 256, [1, 1], false),
            rec_block(256, 256, [1, 1], false),
            rec_block(256, 256, [1, 1], true),
        ],
        vec![
            rec_block(256, 512, [2, 1], false),
            rec_block(512, 512, [1, 1], true),
            rec_block(512, 512, [1, 1], false),
            rec_block(512, 512, [1, 1], true),
            rec_block(512, 512, [1, 1], false),
            rec_block(512, 512, [1, 1], true),
            rec_block(512, 512, [1, 1], false),
        ],
        vec![
            rec_block(512, 768, [2, 1], false),
            rec_block(768, 768, [1, 1], true),
            rec_block(768, 768, [1, 1], false),
        ],
    ]
}

fn small_recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![rec_block(96, 96, [1, 1], true)],
        vec![
            rec_block(96, 96, [1, 1], false),
            rec_block(96, 96, [1, 1], false),
        ],
        vec![
            rec_block(96, 192, [2, 1], false),
            rec_block(192, 192, [1, 1], true),
            rec_block(192, 192, [1, 1], false),
            rec_block(192, 192, [1, 1], true),
            rec_block(192, 192, [1, 1], false),
            rec_block(192, 192, [1, 1], true),
            rec_block(192, 192, [1, 1], false),
        ],
        vec![
            rec_block(192, 384, [2, 1], false),
            rec_block(384, 384, [1, 1], true),
            rec_block(384, 384, [1, 1], false),
        ],
    ]
}

fn tiny_recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![rec_block(48, 48, [1, 1], true)],
        vec![rec_block(48, 48, [1, 1], false)],
        vec![
            rec_block(48, 96, [2, 1], false),
            rec_block(96, 96, [1, 1], true),
            rec_block(96, 96, [1, 1], false),
        ],
        vec![
            rec_block(96, 160, [2, 1], false),
            rec_block(160, 160, [1, 1], true),
            rec_block(160, 160, [1, 1], false),
            rec_block(160, 160, [1, 1], false),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(height: usize, width: usize) -> Tensor {
        Tensor::new_f32(vec![1, 3, height, width], vec![0.0; 3 * height * width])
    }

    #[test]
    fn validates_model_input_boundaries() {
        assert!(validate_input(&input(32, 32), true).is_ok());
        assert!(validate_input(&input(32, 31), true).is_err());
        assert!(validate_input(&input(48, 5), false).is_ok());
        assert!(validate_input(&input(48, 4), false).is_err());
    }

    #[test]
    fn rank_one_conv_matches_dense_fused_weight() -> Result<()> {
        let (in_channels, out_channels, kernel, height, width) = (4, 5, 9, 11, 19);
        let spatial = kernel * kernel;
        let depthwise = (0..in_channels * spatial)
            .map(|index| ((index * 17 % 43) as f32 - 21.0) / 97.0)
            .collect::<Vec<_>>();
        let pointwise = (0..out_channels * in_channels)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) / 31.0)
            .collect::<Vec<_>>();
        let mut fused = vec![0.0; out_channels * in_channels * spatial];
        for output_channel in 0..out_channels {
            for input_channel in 0..in_channels {
                let scale = pointwise[output_channel * in_channels + input_channel];
                let offset = (output_channel * in_channels + input_channel) * spatial;
                for index in 0..spatial {
                    fused[offset + index] = scale * depthwise[input_channel * spatial + index];
                }
            }
        }
        let weight = Tensor::new_f32(vec![out_channels, in_channels, kernel, kernel], fused);
        let bias = Tensor::new_f32(
            vec![out_channels],
            (0..out_channels)
                .map(|channel| (channel as f32 - 2.0) / 13.0)
                .collect(),
        );
        let dense = BackendConv2d::new(
            weight.clone(),
            Some(bias.clone()),
            [1, 1],
            [kernel / 2; 4],
            1,
        )?;
        let rank_one = RankOneConv2d::from_fused(weight, bias, in_channels, out_channels, kernel)?;
        let input = Tensor::new_f32(
            vec![1, in_channels, height, width],
            (0..in_channels * height * width)
                .map(|index| ((index * 23 % 61) as f32 - 30.0) / 37.0)
                .collect(),
        );

        let expected = dense.forward(&input)?;
        let actual = rank_one.forward(&input)?;
        let maximum_error = expected
            .as_f32()?
            .iter()
            .zip(actual.as_f32()?)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            maximum_error < 2.0e-5,
            "rank-one convolution maximum error: {maximum_error}"
        );
        Ok(())
    }

    #[test]
    fn rank_one_factorization_rejects_nonseparable_weight() {
        let mut values = vec![0.0; 2 * 3 * 3];
        values[0] = 1.0;
        values[3 * 3 + 1] = 1.0;
        let weight = Tensor::new_f32(vec![2, 1, 3, 3], values);
        assert!(factor_rank_one_conv_weight(&weight, 1, 2, 3).is_err());
    }

    #[test]
    fn fuses_intraclass_weights_into_center_axes() -> Result<()> {
        let (out_channels, in_channels, kernel) = (2, 2, 3);
        let symmetric = (0..out_channels * in_channels * kernel * kernel)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let vertical = (0..out_channels * in_channels * kernel)
            .map(|index| 100.0 + index as f32)
            .collect::<Vec<_>>();
        let horizontal = (0..out_channels * in_channels * kernel)
            .map(|index| 200.0 + index as f32)
            .collect::<Vec<_>>();
        let (weight, bias) = fuse_intraclass_conv(
            Tensor::new_f32(
                vec![out_channels, in_channels, kernel, kernel],
                symmetric.clone(),
            ),
            Tensor::new_f32(vec![out_channels], vec![1.0, 2.0]),
            Tensor::new_f32(vec![out_channels, in_channels, kernel, 1], vertical.clone()),
            Tensor::new_f32(vec![out_channels], vec![3.0, 4.0]),
            Tensor::new_f32(
                vec![out_channels, in_channels, 1, kernel],
                horizontal.clone(),
            ),
            Tensor::new_f32(vec![out_channels], vec![5.0, 6.0]),
        )?;

        let weight = weight.as_f32()?;
        let center = kernel / 2;
        for output in 0..out_channels {
            for input in 0..in_channels {
                let branch_offset = (output * in_channels + input) * kernel;
                let symmetric_offset = branch_offset * kernel;
                for row in 0..kernel {
                    for column in 0..kernel {
                        let mut expected = symmetric[symmetric_offset + row * kernel + column];
                        if column == center {
                            expected += vertical[branch_offset + row];
                        }
                        if row == center {
                            expected += horizontal[branch_offset + column];
                        }
                        assert_eq!(weight[symmetric_offset + row * kernel + column], expected);
                    }
                }
            }
        }
        assert_eq!(bias.as_f32()?, [9.0, 12.0]);
        Ok(())
    }

    #[test]
    fn fused_intraclass_conv_matches_three_padded_branches() -> Result<()> {
        let (out_channels, in_channels, kernel) = (2, 3, 5);
        let patterned = |length: usize, multiplier: usize, scale: f32| {
            (0..length)
                .map(|index| ((index * multiplier % 31) as f32 - 15.0) * scale)
                .collect::<Vec<_>>()
        };
        let symmetric_weight = Tensor::new_f32(
            vec![out_channels, in_channels, kernel, kernel],
            patterned(out_channels * in_channels * kernel * kernel, 7, 0.01),
        );
        let vertical_weight = Tensor::new_f32(
            vec![out_channels, in_channels, kernel, 1],
            patterned(out_channels * in_channels * kernel, 11, 0.015),
        );
        let horizontal_weight = Tensor::new_f32(
            vec![out_channels, in_channels, 1, kernel],
            patterned(out_channels * in_channels * kernel, 13, 0.012),
        );
        let symmetric_bias = Tensor::new_f32(vec![out_channels], vec![0.07, -0.03]);
        let vertical_bias = Tensor::new_f32(vec![out_channels], vec![-0.02, 0.05]);
        let horizontal_bias = Tensor::new_f32(vec![out_channels], vec![0.01, -0.04]);
        let padding = kernel / 2;

        let symmetric_conv = BackendConv2d::new(
            symmetric_weight.clone(),
            Some(symmetric_bias.clone()),
            [1, 1],
            [padding; 4],
            1,
        )?;
        let vertical_conv = BackendConv2d::new(
            vertical_weight.clone(),
            Some(vertical_bias.clone()),
            [1, 1],
            [padding, 0, padding, 0],
            1,
        )?;
        let horizontal_conv = BackendConv2d::new(
            horizontal_weight.clone(),
            Some(horizontal_bias.clone()),
            [1, 1],
            [0, padding, 0, padding],
            1,
        )?;
        let (fused_weight, fused_bias) = fuse_intraclass_conv(
            symmetric_weight,
            symmetric_bias,
            vertical_weight,
            vertical_bias,
            horizontal_weight,
            horizontal_bias,
        )?;
        let fused_conv =
            BackendConv2d::new(fused_weight, Some(fused_bias), [1, 1], [padding; 4], 1)?;
        let input = Tensor::new_f32(
            vec![1, in_channels, 7, 8],
            patterned(in_channels * 7 * 8, 17, 0.02),
        );

        let expected = symmetric_conv
            .forward(&input)?
            .into_add(&vertical_conv.forward(&input)?)?
            .into_add(&horizontal_conv.forward(&input)?)?;
        let actual = fused_conv.forward(&input)?;
        assert_eq!(actual.shape(), expected.shape());
        for (index, (actual, expected)) in
            actual.as_f32()?.iter().zip(expected.as_f32()?).enumerate()
        {
            let tolerance = 2e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "output {index} differs: fused={actual}, branches={expected}"
            );
        }
        Ok(())
    }
}
