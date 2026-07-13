use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    self as nn, BatchNorm, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm, Linear, Module,
    ModuleT, VarBuilder,
};
use std::path::Path;
use std::str::FromStr;

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

#[derive(Clone, Copy)]
enum Activation {
    None,
    Relu,
    Silu,
    HardSigmoid,
    HardSigmoidFive,
}

impl Activation {
    fn forward(self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::None => Ok(input.clone()),
            Self::Relu => input.relu(),
            Self::Silu => input.silu(),
            Self::HardSigmoid => input.affine(1.0, 3.0)?.clamp(0.0, 6.0)? / 6.0,
            Self::HardSigmoidFive => input.affine(0.2, 0.5)?.clamp(0.0, 1.0),
        }
    }
}

#[derive(Clone)]
struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: (usize, usize),
    stride: (usize, usize),
    groups: usize,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
        groups: usize,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_channels, in_channels / groups, kernel.0, kernel.1),
            "weight",
        )?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            padding,
            stride,
            groups,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (padding_h, padding_w) = self.padding;
        let (stride_h, stride_w) = self.stride;
        let (_, _, kernel_h, kernel_w) = self.weight.dims4()?;
        if self.groups == 1
            && self.stride == (1, 1)
            && self.padding == (4, 4)
            && (kernel_h, kernel_w) == (9, 9)
        {
            return self.forward_tiled_9x9(input);
        }
        let padded = if padding_h == padding_w {
            input.clone()
        } else {
            input
                .pad_with_zeros(2, padding_h, padding_h)?
                .pad_with_zeros(3, padding_w, padding_w)?
        };
        let padding = if padding_h == padding_w { padding_h } else { 0 };
        let stride = if stride_h == stride_w { stride_h } else { 1 };
        let mut output = padded.conv2d(&self.weight, padding, stride, 1, self.groups)?;

        output = self.add_bias(output)?;

        // Candle's Conv2d currently uses a scalar stride. PP-OCRv6 recognition
        // has two (2, 1) depthwise convolutions, so subsample their height after
        // an otherwise identical stride-1 convolution.
        if stride_h != stride_w {
            output = select_every(&output, 2, stride_h)?;
            output = select_every(&output, 3, stride_w)?;
        }
        Ok(output.detach())
    }

    fn forward_tiled_9x9(&self, input: &Tensor) -> Result<Tensor> {
        const OUTPUT_ROWS_PER_TILE: usize = 8;
        let (_, _, output_height, _) = input.dims4()?;
        let padded = input.pad_with_zeros(2, 4, 4)?.pad_with_zeros(3, 4, 4)?;
        let mut tiles = Vec::with_capacity(output_height.div_ceil(OUTPUT_ROWS_PER_TILE));
        for start in (0..output_height).step_by(OUTPUT_ROWS_PER_TILE) {
            let rows = (output_height - start).min(OUTPUT_ROWS_PER_TILE);
            let tile_input = padded.narrow(2, start, rows + 8)?;
            let output = tile_input.conv2d(&self.weight, 0, 1, 1, 1)?;
            tiles.push(self.add_bias(output)?.detach());
        }
        let refs = tiles.iter().collect::<Vec<_>>();
        Ok(Tensor::cat(&refs, 2)?.detach())
    }

    fn add_bias(&self, output: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => output.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1, 1))?),
            None => Ok(output),
        }
    }
}

fn select_every(input: &Tensor, dim: usize, stride: usize) -> Result<Tensor> {
    if stride == 1 {
        return Ok(input.clone());
    }
    let end = input.dim(dim)? as u32;
    let indices = Tensor::arange_step(0u32, end, stride as u32, input.device())?;
    input.index_select(&indices, dim)
}

struct ConvBnAct {
    conv: Conv2d,
    norm: BatchNorm,
    activation: Activation,
}

impl ConvBnAct {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: VarBuilder<'_>,
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
        groups: usize,
        norm_name: &str,
        activation: Activation,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::load(
                vb.pp("convolution"),
                in_channels,
                out_channels,
                kernel,
                stride,
                padding,
                bias,
                groups,
            )?,
            norm: nn::batch_norm(out_channels, 1e-5, vb.pp(norm_name))?,
            activation,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.conv.forward(input)?;
        let output = self.norm.forward_t(&output, false)?;
        Ok(self.activation.forward(&output)?.detach())
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
                (1, 1),
                (1, 1),
                (0, 0),
                true,
                1,
            )?,
            expand: Conv2d::load(
                vb.pp("convolutions").pp(2),
                channels / 4,
                channels,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
                1,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let pooled = input.mean_keepdim((2, 3))?;
        let reduced = self.reduce.forward(&pooled)?.relu()?;
        let attention = Activation::HardSigmoid.forward(&self.expand.forward(&reduced)?)?;
        input.broadcast_mul(&attention)
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
    stride: (usize, usize),
    use_se: bool,
}

impl LcNetBlock {
    fn load(vb: VarBuilder<'_>, spec: BlockSpec) -> Result<Self> {
        let residual = spec.in_channels == spec.out_channels && spec.stride == (1, 1);
        let token_conv = if residual {
            TokenConv::Direct(Conv2d::load(
                vb.pp("token_conv"),
                spec.in_channels,
                spec.out_channels,
                (spec.kernel, spec.kernel),
                spec.stride,
                (spec.kernel / 2, spec.kernel / 2),
                true,
                spec.in_channels,
            )?)
        } else {
            TokenConv::ConvBn(ConvBnAct::load(
                vb.pp("token_conv"),
                spec.in_channels,
                spec.in_channels,
                (spec.kernel, spec.kernel),
                spec.stride,
                (spec.kernel / 2, spec.kernel / 2),
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
                (1, 1),
                (1, 1),
                (0, 0),
                false,
                1,
                "normalization",
                Activation::None,
            )?,
            channel_conv2: ConvBnAct::load(
                vb.pp("channel_conv2"),
                spec.in_channels * 2,
                spec.out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
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
            Some(se) => se.forward(&output)?,
            None => output,
        };
        let shortcut = output.clone();
        let output = self.channel_conv1.forward(&output)?.gelu_erf()?;
        let output = self.channel_conv2.forward(&output)?;
        if self.residual {
            shortcut.broadcast_add(&output)
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
        let conv = |name, in_channels, out_channels, kernel, stride, padding| {
            ConvBnAct::load(
                vb.pp(name),
                in_channels,
                out_channels,
                kernel,
                stride,
                padding,
                false,
                1,
                "normalization",
                activation,
            )
        };
        Ok(Self {
            stem1: conv("stem1", 3, mid_channels, (3, 3), (2, 2), (1, 1))?,
            stem2a: conv(
                "stem2a",
                mid_channels,
                mid_channels / 2,
                (2, 2),
                (1, 1),
                (0, 0),
            )?,
            stem2b: conv(
                "stem2b",
                mid_channels / 2,
                mid_channels,
                (2, 2),
                (1, 1),
                (0, 0),
            )?,
            stem3: conv(
                "stem3",
                mid_channels * 2,
                mid_channels,
                (3, 3),
                (2, 2),
                (1, 1),
            )?,
            stem4: conv("stem4", mid_channels, out_channels, (1, 1), (1, 1), (0, 0))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let embedding = self.stem1.forward(input)?;
        let embedding = embedding.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        let branch = self.stem2a.forward(&embedding)?;
        let branch = branch.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        let branch = self.stem2b.forward(&branch)?;
        let pooled = embedding.max_pool2d_with_stride((2, 2), (1, 1))?;
        let merged = Tensor::cat(&[&pooled, &branch], 1)?;
        let output = self.stem3.forward(&merged)?;
        self.stem4.forward(&output)
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
                (3, 3),
                (2, 2),
                (1, 1),
                false,
                1,
                "normalization",
                Activation::None,
            )?,
            conv2: ConvBnAct::load(
                vb.pp("conv2"),
                mid_channels,
                out_channels,
                (3, 3),
                (2, 2),
                (1, 1),
                false,
                1,
                "normalization",
                Activation::None,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.conv2.forward(&self.conv1.forward(input)?.gelu_erf()?)
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
    Large(LargeStem),
    Small(SmallStem),
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
            } => LcNetStem::Large(LargeStem::load(
                vb.pp("convolution"),
                mid_channels,
                out_channels,
                activation,
            )?),
            StemSpec::Small {
                mid_channels,
                out_channels,
            } => LcNetStem::Small(SmallStem::load(
                vb.pp("convolution"),
                mid_channels,
                out_channels,
            )?),
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
    vertical_long: Conv2d,
    vertical_mid: Conv2d,
    vertical_short: Conv2d,
    horizontal_long: Conv2d,
    horizontal_mid: Conv2d,
    horizontal_short: Conv2d,
    symmetric_long: Conv2d,
    symmetric_mid: Conv2d,
    symmetric_short: Conv2d,
    final_conv: ConvBnAct,
}

impl IntraclassBlock {
    fn load(vb: VarBuilder<'_>) -> Result<Self> {
        let regular = |name, kernel, padding| {
            Conv2d::load(vb.pp(name), 32, 32, kernel, (1, 1), padding, true, 1)
        };
        Ok(Self {
            reduce: Conv2d::load(
                vb.pp("conv_reduce_channel"),
                64,
                32,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
                1,
            )?,
            vertical_long: regular("vertical_long_to_small_conv_longratio", (7, 1), (3, 0))?,
            vertical_mid: regular("vertical_long_to_small_conv_midratio", (5, 1), (2, 0))?,
            vertical_short: regular("vertical_long_to_small_conv_shortratio", (3, 1), (1, 0))?,
            horizontal_long: regular("horizontal_small_to_long_conv_longratio", (1, 7), (0, 3))?,
            horizontal_mid: regular("horizontal_small_to_long_conv_midratio", (1, 5), (0, 2))?,
            horizontal_short: regular("horizontal_small_to_long_conv_shortratio", (1, 3), (0, 1))?,
            symmetric_long: regular("symmetric_conv_long_longratio", (7, 7), (3, 3))?,
            symmetric_mid: regular("symmetric_conv_long_midratio", (5, 5), (2, 2))?,
            symmetric_short: regular("symmetric_conv_long_shortratio", (3, 3), (1, 1))?,
            final_conv: ConvBnAct::load(
                vb.pp("conv_final"),
                32,
                64,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
                1,
                "norm",
                Activation::Relu,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let reduced = self.reduce.forward(input)?;
        let layer7 = self
            .symmetric_long
            .forward(&reduced)?
            .broadcast_add(&self.vertical_long.forward(&reduced)?)?
            .broadcast_add(&self.horizontal_long.forward(&reduced)?)?;
        let layer5 = self
            .symmetric_mid
            .forward(&layer7)?
            .broadcast_add(&self.vertical_mid.forward(&layer7)?)?
            .broadcast_add(&self.horizontal_mid.forward(&layer7)?)?;
        let layer3 = self
            .symmetric_short
            .forward(&layer5)?
            .broadcast_add(&self.vertical_short.forward(&layer5)?)?
            .broadcast_add(&self.horizontal_short.forward(&layer5)?)?;
        input.broadcast_add(&self.final_conv.forward(&layer3)?)
    }
}

struct DetectorNeck {
    adjust: Vec<Conv2d>,
    project: Vec<Conv2d>,
    pan_head: Vec<Conv2d>,
    pan_lateral: Vec<Conv2d>,
    intraclass: Vec<IntraclassBlock>,
}

impl DetectorNeck {
    fn load(vb: VarBuilder<'_>) -> Result<Self> {
        let stage_channels = [128, 256, 512, 896];
        let adjust = stage_channels
            .iter()
            .enumerate()
            .map(|(index, channels)| {
                Conv2d::load(
                    vb.pp("input_channel_adjustment_convolution").pp(index),
                    *channels,
                    256,
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    false,
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let project = (0..4)
            .map(|index| {
                Conv2d::load(
                    vb.pp("input_feature_projection_convolution").pp(index),
                    256,
                    64,
                    (9, 9),
                    (1, 1),
                    (4, 4),
                    true,
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let pan_head = (0..3)
            .map(|index| {
                Conv2d::load(
                    vb.pp("path_aggregation_head_convolution").pp(index),
                    64,
                    64,
                    (3, 3),
                    (2, 2),
                    (1, 1),
                    false,
                    1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let pan_lateral = (0..4)
            .map(|index| {
                Conv2d::load(
                    vb.pp("path_aggregation_lateral_convolution").pp(index),
                    64,
                    64,
                    (9, 9),
                    (1, 1),
                    (4, 4),
                    true,
                    1,
                )
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

        let mut top_down = [None, None, None, Some(adjusted[3].clone())];
        for index in (0..3).rev() {
            let upper = top_down[index + 1].as_ref().expect("upper feature exists");
            top_down[index] = Some(adjusted[index].broadcast_add(&upsample(upper, 2)?)?);
        }

        let mut projected = Vec::with_capacity(4);
        for (index, top_feature) in top_down.iter().enumerate() {
            let source = if index == 3 {
                &adjusted[3]
            } else {
                top_feature.as_ref().expect("top-down feature exists")
            };
            projected.push(self.project[index].forward(source)?);
        }

        let mut bottom_up = [Some(projected[0].clone()), None, None, None];
        for (index, projection) in projected.iter().enumerate().skip(1) {
            let lower = bottom_up[index - 1].as_ref().expect("lower feature exists");
            bottom_up[index] =
                Some(projection.broadcast_add(&self.pan_head[index - 1].forward(lower)?)?);
        }

        let lateral = (0..4)
            .map(|index| {
                let source = if index == 0 {
                    &projected[0]
                } else {
                    bottom_up[index].as_ref().expect("bottom-up feature exists")
                };
                self.pan_lateral[index].forward(source)
            })
            .collect::<Result<Vec<_>>>()?;
        let refined = self
            .intraclass
            .iter()
            .zip(lateral.iter())
            .map(|(block, feature)| block.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        let scales = [1, 2, 4, 8];
        let mut features = refined
            .iter()
            .zip(scales)
            .map(|(feature, scale)| upsample(feature, scale))
            .collect::<Result<Vec<_>>>()?;
        features.reverse();
        let refs = features.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 1)
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
                (1, 1),
                (1, 1),
                (0, 0),
                true,
                1,
            )?,
            expand: Conv2d::load(
                vb.pp("conv2"),
                channels / reduction,
                channels,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
                1,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let pooled = input.mean_keepdim((2, 3))?;
        let hidden = self.reduce.forward(&pooled)?.relu()?;
        let attention = Activation::HardSigmoidFive.forward(&self.expand.forward(&hidden)?)?;
        input.broadcast_mul(&attention)
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
                (1, 1),
                (1, 1),
                (0, 0),
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
        hidden.broadcast_add(&self.squeeze_excitation.forward(&hidden)?)
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
        Ok(Self {
            depthwise: Conv2d::load(
                vb.pp("depthwise_convolution"),
                channels,
                channels,
                (kernel_size, kernel_size),
                (1, 1),
                (kernel_size / 2, kernel_size / 2),
                true,
                channels,
            )?,
            pointwise: Conv2d::load(
                vb.pp("pointwise_convolution"),
                channels,
                channels / 4,
                (1, 1),
                (1, 1),
                (0, 0),
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
        hidden.broadcast_add(&self.squeeze_excitation.forward(&hidden)?)
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
            fused[index] = fused[index].broadcast_add(&upper)?;
        }
        let mut features = self
            .input
            .iter()
            .zip(fused.iter())
            .map(|(conv, feature)| conv.forward(feature))
            .collect::<Result<Vec<_>>>()?;
        for (feature, scale) in features.iter_mut().zip([1, 2, 4, 8]) {
            *feature = upsample(feature, scale)?;
        }
        features.reverse();
        let refs = features.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 1)
    }
}

fn upsample(input: &Tensor, scale: usize) -> Result<Tensor> {
    if scale == 1 {
        return Ok(input.clone());
    }
    let (_, _, height, width) = input.dims4()?;
    input.upsample_nearest2d(height * scale, width * scale)
}

struct TransposeConvBnRelu {
    convolution: ConvTranspose2d,
    norm: BatchNorm,
}

impl TransposeConvBnRelu {
    fn load(vb: VarBuilder<'_>, in_channels: usize, out_channels: usize) -> Result<Self> {
        let weight = vb
            .pp("convolution")
            .get((in_channels, out_channels, 2, 2), "weight")?;
        let bias = vb.pp("convolution").get(out_channels, "bias")?;
        let convolution = ConvTranspose2d::new(
            weight,
            Some(bias),
            ConvTranspose2dConfig {
                stride: 2,
                ..Default::default()
            },
        );
        Ok(Self {
            convolution,
            norm: nn::batch_norm(out_channels, 1e-5, vb.pp("norm"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.norm
            .forward_t(&self.convolution.forward(input)?, false)?
            .relu()
    }
}

struct DetectorHead {
    down: ConvBnAct,
    up: TransposeConvBnRelu,
    final_conv: ConvTranspose2d,
}

impl DetectorHead {
    fn load(vb: VarBuilder<'_>, input_channels: usize) -> Result<Self> {
        let hidden_channels = input_channels / 4;
        let down = ConvBnAct::load(
            vb.pp("conv_down"),
            input_channels,
            hidden_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            false,
            1,
            "norm",
            Activation::Relu,
        )?;
        let up = TransposeConvBnRelu::load(vb.pp("conv_up"), hidden_channels, hidden_channels)?;
        let weight = vb
            .pp("conv_final")
            .get((hidden_channels, 1, 2, 2), "weight")?;
        let bias = vb.pp("conv_final").get(1, "bias")?;
        let final_conv = ConvTranspose2d::new(
            weight,
            Some(bias),
            ConvTranspose2dConfig {
                stride: 2,
                ..Default::default()
            },
        );
        Ok(Self {
            down,
            up,
            final_conv,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.down.forward(input)?;
        let output = self.up.forward(&output)?;
        nn::ops::sigmoid(&self.final_conv.forward(&output)?)
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
}

impl Detector {
    pub fn load(path: impl AsRef<Path>, device: &Device, size: ModelSize) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }?;
        let encoder = vb.pp("model").pp("backbone").pp("encoder");
        match size {
            ModelSize::Medium => Ok(Self {
                backbone: LcNetBackbone::load(
                    encoder,
                    &detector_stages(),
                    StemSpec::Large {
                        mid_channels: 64,
                        out_channels: 128,
                    },
                    Activation::Relu,
                )?,
                neck: DetectorNeckKind::Medium(DetectorNeck::load(vb.pp("model").pp("neck"))?),
                head: DetectorHead::load(vb.pp("head"), 256)?,
            }),
            ModelSize::Small => Ok(Self {
                backbone: LcNetBackbone::load(
                    encoder,
                    &detector_stages_for_channels([48, 96, 192, 384]),
                    StemSpec::Large {
                        mid_channels: 24,
                        out_channels: 48,
                    },
                    Activation::Relu,
                )?,
                neck: DetectorNeckKind::RepLkFpn(RepLkFpn::load(
                    vb.pp("model").pp("neck"),
                    [48, 96, 192, 384],
                    96,
                    7,
                )?),
                head: DetectorHead::load(vb.pp("head"), 96)?,
            }),
            ModelSize::Tiny => Ok(Self {
                backbone: LcNetBackbone::load(
                    encoder,
                    &detector_stages_for_channels([32, 48, 64, 160]),
                    StemSpec::Large {
                        mid_channels: 16,
                        out_channels: 32,
                    },
                    Activation::Relu,
                )?,
                neck: DetectorNeckKind::RepLkFpn(RepLkFpn::load(
                    vb.pp("model").pp("neck"),
                    [32, 48, 64, 160],
                    64,
                    5,
                )?),
                head: DetectorHead::load(vb.pp("head"), 64)?,
            }),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;
        self.head.forward(&self.neck.forward(&features)?)
    }
}

fn detector_stages() -> Vec<Vec<BlockSpec>> {
    detector_stages_for_channels([128, 256, 512, 896])
}

fn detector_stages_for_channels(channels: [usize; 4]) -> Vec<Vec<BlockSpec>> {
    let [stage1, stage2, stage3, stage4] = channels;
    vec![
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: stage1,
                out_channels: stage1,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage1,
                out_channels: stage1,
                stride: (1, 1),
                use_se: false,
            },
        ],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: stage1,
                out_channels: stage2,
                stride: (2, 2),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage2,
                out_channels: stage2,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage2,
                out_channels: stage2,
                stride: (1, 1),
                use_se: false,
            },
        ],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: stage2,
                out_channels: stage3,
                stride: (2, 2),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage3,
                out_channels: stage3,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage3,
                out_channels: stage3,
                stride: (1, 1),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage3,
                out_channels: stage3,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage3,
                out_channels: stage3,
                stride: (1, 1),
                use_se: false,
            },
        ],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: stage3,
                out_channels: stage4,
                stride: (2, 2),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage4,
                out_channels: stage4,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: stage4,
                out_channels: stage4,
                stride: (1, 1),
                use_se: false,
            },
        ],
    ]
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
            qkv: nn::linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?,
            projection: nn::linear(hidden_size, hidden_size, vb.pp("projection"))?,
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
            .reshape((batch, sequence, 3, self.num_heads, head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let parts = qkv.chunk(3, 0)?;
        let query = parts[0].squeeze(0)?.contiguous()?;
        let key = parts[1].squeeze(0)?.contiguous()?;
        let value = parts[2].squeeze(0)?.contiguous()?;
        let weights = query
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            .affine((head_dim as f64).powf(-0.5), 0.0)?;
        let weights = nn::ops::softmax_last_dim(&weights)?;
        let output = weights
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, sequence, self.hidden_size))?;
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
            fc1: nn::linear(hidden_size, mlp_size, vb.pp("fc1"))?,
            fc2: nn::linear(mlp_size, hidden_size, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(input)?.silu()?)
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
            layer_norm1: nn::layer_norm(hidden_size, 1e-6, vb.pp("layer_norm1"))?,
            attention: RecAttention::load(vb.pp("self_attn"), hidden_size, num_heads)?,
            layer_norm2: nn::layer_norm(hidden_size, 1e-6, vb.pp("layer_norm2"))?,
            mlp: RecMlp::load(vb.pp("mlp"), hidden_size, mlp_size)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output =
            input.broadcast_add(&self.attention.forward(&self.layer_norm1.forward(input)?)?)?;
        output.broadcast_add(&self.mlp.forward(&self.layer_norm2.forward(&output)?)?)
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
        let load_conv = |index, in_channels, out_channels, kernel, groups| {
            ConvBnAct::load(
                vb.pp("conv_block").pp(index),
                in_channels,
                out_channels,
                kernel,
                (1, 1),
                (kernel.0 / 2, kernel.1 / 2),
                false,
                groups,
                "normalization",
                Activation::Silu,
            )
        };
        Ok(Self {
            skip: load_conv(0, input_channels, hidden_size, (1, 1), 1)?,
            reduce: load_conv(1, input_channels, hidden_size, (1, 1), 1)?,
            local: load_conv(2, hidden_size, hidden_size, (1, 7), hidden_size)?,
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
            norm: nn::layer_norm(hidden_size, 1e-6, vb.pp("norm"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let residual = self.skip.forward(input)?;
        let hidden = self.reduce.forward(input)?;
        let hidden = hidden.broadcast_add(&self.local.forward(&hidden)?)?;
        let (batch, channels, height, width) = hidden.dims4()?;
        let mut hidden = hidden.flatten(2, 3)?.transpose(1, 2)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden
            .reshape((batch, height, width, channels))?
            .permute((0, 3, 1, 2))?;
        let hidden = hidden.broadcast_add(&residual)?;
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
            classifier: nn::linear(hidden_size, classes, vb.pp("head"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        nn::ops::softmax_last_dim(&self.classifier.forward(&self.encoder.forward(input)?)?)
    }
}

struct Conv1d {
    weight: Tensor,
    padding: usize,
    groups: usize,
}

impl Conv1d {
    fn load(
        vb: VarBuilder<'_>,
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = vb
            .get(
                (output_channels, input_channels / groups, kernel_size),
                "weight",
            )?
            .reshape((output_channels, input_channels / groups, 1, kernel_size))?;
        Ok(Self {
            weight,
            padding: kernel_size / 2,
            groups,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input = input
            .unsqueeze(2)?
            .pad_with_zeros(3, self.padding, self.padding)?;
        input.conv2d(&self.weight, 0, 1, 1, self.groups)?.squeeze(2)
    }
}

fn hard_swish(input: &Tensor) -> Result<Tensor> {
    let gate = input.affine(1.0, 3.0)?.clamp(0.0, 6.0)?;
    input.broadcast_mul(&gate)? / 6.0
}

struct TinyRecognizerHead {
    conv1: Conv1d,
    norm1: BatchNorm,
    conv2: Conv1d,
    norm2: BatchNorm,
    fc1: Linear,
    fc2: Linear,
}

impl TinyRecognizerHead {
    fn load(vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            conv1: Conv1d::load(vb.pp("conv1"), 160, 160, 5, 160)?,
            norm1: nn::batch_norm(160, 1e-5, vb.pp("norm1"))?,
            conv2: Conv1d::load(vb.pp("conv2"), 160, 160, 1, 1)?,
            norm2: nn::batch_norm(160, 1e-5, vb.pp("norm2"))?,
            fc1: nn::linear(160, 80, vb.pp("fc1"))?,
            fc2: nn::linear(80, 6_906, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = input.squeeze(2)?;
        let hidden = self
            .norm1
            .forward_t(&self.conv1.forward(&hidden)?.unsqueeze(2)?, false)?
            .squeeze(2)?;
        let hidden = hard_swish(&hidden)?;
        let hidden = self
            .norm2
            .forward_t(&self.conv2.forward(&hidden)?.unsqueeze(2)?, false)?
            .squeeze(2)?;
        let hidden = hard_swish(&hidden)?.transpose(1, 2)?;
        nn::ops::softmax_last_dim(&self.fc2.forward(&self.fc1.forward(&hidden)?)?)
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
}

impl Recognizer {
    pub fn load(path: impl AsRef<Path>, device: &Device, size: ModelSize) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }?;
        let encoder = vb.pp("model").pp("backbone").pp("encoder");
        match size {
            ModelSize::Medium => Ok(Self {
                backbone: LcNetBackbone::load(
                    encoder,
                    &recognizer_stages(),
                    StemSpec::Large {
                        mid_channels: 64,
                        out_channels: 128,
                    },
                    Activation::Relu,
                )?,
                head: RecognizerHeadKind::LightSvtr(LightSvtrRecognizerHead::load(
                    vb.pp("head"),
                    768,
                    192,
                    768,
                    18_710,
                )?),
            }),
            ModelSize::Small => Ok(Self {
                backbone: LcNetBackbone::load(
                    encoder,
                    &small_recognizer_stages(),
                    StemSpec::Large {
                        mid_channels: 48,
                        out_channels: 96,
                    },
                    Activation::Relu,
                )?,
                head: RecognizerHeadKind::LightSvtr(LightSvtrRecognizerHead::load(
                    vb.pp("head"),
                    384,
                    120,
                    240,
                    18_710,
                )?),
            }),
            ModelSize::Tiny => Ok(Self {
                backbone: LcNetBackbone::load(
                    encoder,
                    &tiny_recognizer_stages(),
                    StemSpec::Small {
                        mid_channels: 24,
                        out_channels: 48,
                    },
                    Activation::Relu,
                )?,
                head: RecognizerHeadKind::Tiny(TinyRecognizerHead::load(vb.pp("head"))?),
            }),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;
        let feature = features.last().expect("recognizer backbone has stages");
        let pooled = feature.avg_pool2d_with_stride((3, 2), (3, 2))?;
        self.head.forward(&pooled)
    }
}

fn recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![BlockSpec {
            kernel: 3,
            in_channels: 128,
            out_channels: 128,
            stride: (1, 1),
            use_se: true,
        }],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: 128,
                out_channels: 256,
                stride: (1, 1),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 256,
                out_channels: 256,
                stride: (1, 1),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 256,
                out_channels: 256,
                stride: (1, 1),
                use_se: true,
            },
        ],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: 256,
                out_channels: 512,
                stride: (2, 1),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 512,
                out_channels: 512,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 512,
                out_channels: 512,
                stride: (1, 1),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 512,
                out_channels: 512,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 512,
                out_channels: 512,
                stride: (1, 1),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 512,
                out_channels: 512,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 512,
                out_channels: 512,
                stride: (1, 1),
                use_se: false,
            },
        ],
        vec![
            BlockSpec {
                kernel: 3,
                in_channels: 512,
                out_channels: 768,
                stride: (2, 1),
                use_se: false,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 768,
                out_channels: 768,
                stride: (1, 1),
                use_se: true,
            },
            BlockSpec {
                kernel: 3,
                in_channels: 768,
                out_channels: 768,
                stride: (1, 1),
                use_se: false,
            },
        ],
    ]
}

fn rec_block(
    in_channels: usize,
    out_channels: usize,
    stride: (usize, usize),
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

fn small_recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![rec_block(96, 96, (1, 1), true)],
        vec![
            rec_block(96, 96, (1, 1), false),
            rec_block(96, 96, (1, 1), false),
        ],
        vec![
            rec_block(96, 192, (2, 1), false),
            rec_block(192, 192, (1, 1), true),
            rec_block(192, 192, (1, 1), false),
            rec_block(192, 192, (1, 1), true),
            rec_block(192, 192, (1, 1), false),
            rec_block(192, 192, (1, 1), true),
            rec_block(192, 192, (1, 1), false),
        ],
        vec![
            rec_block(192, 384, (2, 1), false),
            rec_block(384, 384, (1, 1), true),
            rec_block(384, 384, (1, 1), false),
        ],
    ]
}

fn tiny_recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![rec_block(48, 48, (1, 1), true)],
        vec![rec_block(48, 48, (1, 1), false)],
        vec![
            rec_block(48, 96, (2, 1), false),
            rec_block(96, 96, (1, 1), true),
            rec_block(96, 96, (1, 1), false),
        ],
        vec![
            rec_block(96, 160, (2, 1), false),
            rec_block(160, 160, (1, 1), true),
            rec_block(160, 160, (1, 1), false),
            rec_block(160, 160, (1, 1), false),
        ],
    ]
}
