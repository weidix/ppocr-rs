use super::error::{Error, Result};
use super::runtime::{
    Activation, ConvDesc, Gpu, GpuImage, GraphBuilder, ImagePreprocess, Session, Value,
};
use super::weights::Weights;
use crate::models::ModelSize;
use std::path::Path;

const BN_EPSILON: f32 = 1.0e-5;
const MEDIUM_DETECTOR_TENSORS: usize = 350;
const SMALL_DETECTOR_TENSORS: usize = 271;
const TINY_DETECTOR_TENSORS: usize = 271;
const MEDIUM_RECOGNIZER_TENSORS: usize = 269;
const SMALL_RECOGNIZER_TENSORS: usize = 250;
const TINY_RECOGNIZER_TENSORS: usize = 150;
const LARGE_RECOGNIZER_CLASSES: usize = 18_710;
const TINY_RECOGNIZER_CLASSES: usize = 6_906;

impl ModelSize {
    const fn detector_tensors(self) -> usize {
        match self {
            Self::Medium => MEDIUM_DETECTOR_TENSORS,
            Self::Small => SMALL_DETECTOR_TENSORS,
            Self::Tiny => TINY_DETECTOR_TENSORS,
        }
    }

    const fn recognizer_tensors(self) -> usize {
        match self {
            Self::Medium => MEDIUM_RECOGNIZER_TENSORS,
            Self::Small => SMALL_RECOGNIZER_TENSORS,
            Self::Tiny => TINY_RECOGNIZER_TENSORS,
        }
    }
}

#[derive(Debug)]
pub struct ModelOutput {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub struct Detector {
    session: Session,
    output_shape: [usize; 4],
}

impl Detector {
    pub fn load(
        gpu: &Gpu,
        path: impl AsRef<Path>,
        size: ModelSize,
        input_shape: [usize; 4],
    ) -> Result<Self> {
        validate_detector_shape(input_shape)?;
        let weights = Weights::load(path)?;
        let expected_tensors = size.detector_tensors();
        if weights.len() != expected_tensors {
            return Err(Error::InvalidModel(format!(
                "{} detector has {} tensors; expected {expected_tensors}",
                size.as_str(),
                weights.len(),
            )));
        }

        let (graph, input) = GraphBuilder::new(input_shape)?;
        let mut builder = ModelBuilder::new(graph, &weights);
        let stages = build_detector_backbone(&mut builder, input, size)?;
        let neck = build_detector_neck(&mut builder, &stages, size)?;
        let output = build_detector_head(&mut builder, neck, detector_neck_channels(size))?;
        let output_shape = [
            output.shape.n,
            output.shape.c,
            output.shape.h,
            output.shape.w,
        ];
        if output_shape != [input_shape[0], 1, input_shape[2], input_shape[3]] {
            return Err(Error::InvalidModel(format!(
                "{} detector graph produced {output_shape:?}; expected [{}, 1, {}, {}]",
                size.as_str(),
                input_shape[0],
                input_shape[2],
                input_shape[3]
            )));
        }
        let (packed_weights, plan) = builder.finish(output)?;
        let session = gpu.create_session(packed_weights, plan)?;
        Ok(Self {
            session,
            output_shape,
        })
    }

    pub fn forward(&self, input: &[f32]) -> Result<ModelOutput> {
        let output = self.session.run_nchw(input)?;
        let found = [
            output.shape.n,
            output.shape.c,
            output.shape.h,
            output.shape.w,
        ];
        if found != self.output_shape {
            return Err(Error::Gpu(format!(
                "detector returned shape {found:?}; expected {:?}",
                self.output_shape
            )));
        }
        Ok(ModelOutput {
            shape: self.output_shape.to_vec(),
            values: output.values,
        })
    }

    /// Runs a detector from a device-resident source image.
    pub fn forward_image(
        &self,
        image: &GpuImage,
        preprocess: ImagePreprocess,
    ) -> Result<ModelOutput> {
        let output = self.session.run_image(image, preprocess)?;
        let found = [
            output.shape.n,
            output.shape.c,
            output.shape.h,
            output.shape.w,
        ];
        if found != self.output_shape {
            return Err(Error::Gpu(format!(
                "detector returned shape {found:?}; expected {:?}",
                self.output_shape
            )));
        }
        Ok(ModelOutput {
            shape: self.output_shape.to_vec(),
            values: output.values,
        })
    }

    pub fn benchmark(
        &self,
        input: &[f32],
        warmup: usize,
        runs: usize,
    ) -> Result<Vec<std::time::Duration>> {
        self.session.benchmark_nchw(input, warmup, runs)
    }
}

pub struct Recognizer {
    session: Session,
    output_shape: [usize; 3],
}

impl Recognizer {
    pub fn load(
        gpu: &Gpu,
        path: impl AsRef<Path>,
        size: ModelSize,
        input_shape: [usize; 4],
    ) -> Result<Self> {
        validate_recognizer_shape(input_shape)?;
        let weights = Weights::load(path)?;
        let expected_tensors = size.recognizer_tensors();
        if weights.len() != expected_tensors {
            return Err(Error::InvalidModel(format!(
                "{} recognizer has {} tensors; expected {expected_tensors}",
                size.as_str(),
                weights.len(),
            )));
        }

        let (graph, input) = GraphBuilder::new(input_shape)?;
        let mut builder = ModelBuilder::new(graph, &weights);
        let stages = build_recognizer_backbone(&mut builder, input, size)?;
        let feature = stages
            .last()
            .cloned()
            .ok_or_else(|| Error::InvalidModel("recognizer backbone has no stages".into()))?;
        let pooled = builder.graph.avg_pool(feature, [3, 2], [3, 2])?;
        let backbone_channels = recognizer_backbone_channels(size);
        if pooled.shape.h != 1 || pooled.shape.c != backbone_channels {
            return Err(Error::InvalidModel(format!(
                "{} recognizer pooling produced {:?}; expected height 1 and {backbone_channels} channels",
                size.as_str(),
                pooled.shape,
            )));
        }
        let output = build_recognizer_head(&mut builder, pooled, size)?;
        let output_shape = [output.shape.n, output.shape.w, output.shape.c];
        let classes = size.recognizer_classes();
        if output.shape.h != 1 || output.shape.n != input_shape[0] || output.shape.c != classes {
            return Err(Error::InvalidModel(format!(
                "{} recognizer graph produced {:?}; expected [N, 1, T, {classes}] NHWC",
                size.as_str(),
                output.shape,
            )));
        }
        let (packed_weights, plan) = builder.finish(output)?;
        let session = gpu.create_session(packed_weights, plan)?;
        Ok(Self {
            session,
            output_shape,
        })
    }

    pub fn forward(&self, input: &[f32]) -> Result<ModelOutput> {
        let output = self.session.run_nchw(input)?;
        let found = [output.shape.n, output.shape.w, output.shape.c];
        if output.shape.h != 1 || found != self.output_shape {
            return Err(Error::Gpu(format!(
                "recognizer returned NHWC shape {:?}; expected NTC {:?}",
                output.shape, self.output_shape
            )));
        }
        // RawOutput is compacted in NHWC order. With H=1 this is already N-T-C.
        Ok(ModelOutput {
            shape: self.output_shape.to_vec(),
            values: output.values,
        })
    }

    /// Runs a recognizer from a device-resident source image.
    pub fn forward_image(
        &self,
        image: &GpuImage,
        preprocess: ImagePreprocess,
    ) -> Result<ModelOutput> {
        let output = self.session.run_image(image, preprocess)?;
        let found = [output.shape.n, output.shape.w, output.shape.c];
        if output.shape.h != 1 || found != self.output_shape {
            return Err(Error::Gpu(format!(
                "recognizer returned NHWC shape {:?}; expected NTC {:?}",
                output.shape, self.output_shape
            )));
        }
        Ok(ModelOutput {
            shape: self.output_shape.to_vec(),
            values: output.values,
        })
    }

    pub fn benchmark(
        &self,
        input: &[f32],
        warmup: usize,
        runs: usize,
    ) -> Result<Vec<std::time::Duration>> {
        self.session.benchmark_nchw(input, warmup, runs)
    }
}

fn validate_detector_shape([n, c, h, w]: [usize; 4]) -> Result<()> {
    if n == 0 || c != 3 || h < 32 || w < 32 || h % 32 != 0 || w % 32 != 0 {
        return Err(Error::InvalidInput(format!(
            "detector expects [N, 3, H, W] with N > 0, H/W >= 32 and divisible by 32; got [{n}, {c}, {h}, {w}]"
        )));
    }
    checked_elements([n, c, h, w])?;
    Ok(())
}

fn validate_recognizer_shape([n, c, h, w]: [usize; 4]) -> Result<()> {
    if n == 0 || c != 3 || h != 48 || w < 32 || w % 4 != 0 {
        return Err(Error::InvalidInput(format!(
            "recognizer expects [N, 3, 48, W] with N > 0, W >= 32 and divisible by 4; got [{n}, {c}, {h}, {w}]"
        )));
    }
    checked_elements([n, c, h, w])?;
    Ok(())
}

fn checked_elements(shape: [usize; 4]) -> Result<usize> {
    shape
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| Error::InvalidInput(format!("input shape {shape:?} overflows")))
}

#[derive(Clone, Copy)]
enum SourceLayout {
    Oihw,
    Oiw,
    Oi,
    Iohw,
}

#[derive(Default)]
struct WeightPacker {
    values: Vec<f32>,
}

impl WeightPacker {
    fn push(&mut self, values: &[f32]) -> Result<u32> {
        while !self.values.len().is_multiple_of(4) {
            self.values.push(0.0);
        }
        let offset = u32::try_from(self.values.len())
            .map_err(|_| Error::InvalidModel("packed weights exceed u32 indexing".into()))?;
        self.values.extend_from_slice(values);
        Ok(offset)
    }

    fn finish(mut self) -> Vec<f32> {
        while !self.values.len().is_multiple_of(4) {
            self.values.push(0.0);
        }
        self.values
    }
}

struct ModelBuilder<'a> {
    graph: GraphBuilder,
    source: &'a Weights,
    packed: WeightPacker,
}

impl<'a> ModelBuilder<'a> {
    fn new(graph: GraphBuilder, source: &'a Weights) -> Self {
        Self {
            graph,
            source,
            packed: WeightPacker::default(),
        }
    }

    fn finish(self, output: Value) -> Result<(Vec<f32>, super::runtime::Plan)> {
        let plan = self.graph.finish(output)?;
        Ok((self.packed.finish(), plan))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        input: Value,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        bias: bool,
        norm: Option<&str>,
        activation: Activation,
    ) -> Result<Value> {
        let desc = self.pack_ungrouped(
            prefix,
            input_channels,
            output_channels,
            kernel,
            stride,
            padding,
            bias,
            norm,
            SourceLayout::Oihw,
        )?;
        self.graph.conv(input, &desc, activation)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_output(
        &mut self,
        input: Value,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        bias: bool,
        norm: Option<&str>,
        activation: Activation,
        output_hw: [usize; 2],
    ) -> Result<Value> {
        let desc = self.pack_ungrouped(
            prefix,
            input_channels,
            output_channels,
            kernel,
            stride,
            padding,
            bias,
            norm,
            SourceLayout::Oihw,
        )?;
        self.graph.conv_output(input, &desc, activation, output_hw)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_add(
        &mut self,
        input: Value,
        add: Value,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        norm: Option<&str>,
        activation: Activation,
    ) -> Result<Value> {
        let desc = self.pack_ungrouped(
            prefix,
            input_channels,
            output_channels,
            [1, 1],
            [1, 1],
            [0, 0],
            false,
            norm,
            SourceLayout::Oihw,
        )?;
        self.graph.conv_add(input, &desc, activation, add)
    }

    #[allow(clippy::too_many_arguments)]
    fn depthwise(
        &mut self,
        input: Value,
        prefix: &str,
        channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        bias: bool,
        norm: Option<&str>,
        activation: Activation,
        layout: SourceLayout,
    ) -> Result<Value> {
        let desc = self.pack_depthwise(
            prefix, channels, kernel, stride, padding, bias, norm, layout,
        )?;
        self.graph.depthwise(input, &desc, activation)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv_1d(
        &mut self,
        input: Value,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_width: usize,
        padding_width: usize,
        bias: bool,
        norm: Option<&str>,
        activation: Activation,
    ) -> Result<Value> {
        let desc = self.pack_ungrouped(
            prefix,
            input_channels,
            output_channels,
            [1, kernel_width],
            [1, 1],
            [0, padding_width],
            bias,
            norm,
            SourceLayout::Oiw,
        )?;
        self.graph.conv(input, &desc, activation)
    }

    fn linear(
        &mut self,
        input: Value,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        activation: Activation,
    ) -> Result<Value> {
        let desc = self.pack_ungrouped(
            prefix,
            input_channels,
            output_channels,
            [1, 1],
            [1, 1],
            [0, 0],
            true,
            None,
            SourceLayout::Oi,
        )?;
        self.graph.conv(input, &desc, activation)
    }

    fn layer_norm(&mut self, input: Value, prefix: &str, channels: usize) -> Result<Value> {
        if input.shape.c != channels {
            return Err(Error::InvalidModel(format!(
                "layer norm {prefix:?} expects {channels} channels, found {}",
                input.shape.c
            )));
        }
        let weight = read_finite(self.source, &format!("{prefix}.weight"), &[channels])?;
        let bias = read_finite(self.source, &format!("{prefix}.bias"), &[channels])?;
        let weight_offset = self.packed.push(&weight)?;
        let bias_offset = self.packed.push(&bias)?;
        self.graph
            .layer_norm(input, weight_offset, bias_offset, 1.0e-6)
    }

    fn deconv(
        &mut self,
        input: Value,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        norm: Option<&str>,
        activation: Activation,
    ) -> Result<Value> {
        let desc = self.pack_ungrouped(
            prefix,
            input_channels,
            output_channels,
            [2, 2],
            [2, 2],
            [0, 0],
            true,
            norm,
            SourceLayout::Iohw,
        )?;
        self.graph.deconv(input, &desc, activation)
    }

    fn detector_head(
        &mut self,
        input: Value,
        input_channels: usize,
        hidden_channels: usize,
    ) -> Result<Value> {
        let conv = self.pack_ungrouped(
            "head.conv_down.convolution",
            input_channels,
            hidden_channels,
            [3, 3],
            [1, 1],
            [1, 1],
            false,
            Some("head.conv_down.norm"),
            SourceLayout::Oihw,
        )?;
        let up = self.pack_ungrouped(
            "head.conv_up.convolution",
            hidden_channels,
            hidden_channels,
            [2, 2],
            [2, 2],
            [0, 0],
            true,
            Some("head.conv_up.norm"),
            SourceLayout::Iohw,
        )?;
        let final_conv = self.pack_ungrouped(
            "head.conv_final",
            hidden_channels,
            1,
            [2, 2],
            [2, 2],
            [0, 0],
            true,
            None,
            SourceLayout::Iohw,
        )?;
        self.graph
            .fused_detector_head(input, &conv, &up, &final_conv)
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_ungrouped(
        &mut self,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        has_source_bias: bool,
        norm: Option<&str>,
        layout: SourceLayout,
    ) -> Result<ConvDesc> {
        let weight_name = format!("{prefix}.weight");
        let expected = source_shape(layout, input_channels, output_channels, kernel)?;
        let source = read_finite(self.source, &weight_name, &expected)?;
        let bias_name = has_source_bias.then(|| format!("{prefix}.bias"));
        let (scale, bias) =
            folded_channel_params(self.source, bias_name.as_deref(), norm, output_channels)?;
        let output_stride = round_up4(output_channels)?;
        let mut packed = vec![0.0; kernel[0] * kernel[1] * input_channels * output_stride];
        for ky in 0..kernel[0] {
            for kx in 0..kernel[1] {
                for input_channel in 0..input_channels {
                    let target_base =
                        ((ky * kernel[1] + kx) * input_channels + input_channel) * output_stride;
                    for output_channel in 0..output_channels {
                        let source_index = source_index(
                            layout,
                            input_channel,
                            output_channel,
                            ky,
                            kx,
                            input_channels,
                            output_channels,
                            kernel,
                        );
                        packed[target_base + output_channel] =
                            source[source_index] * scale[output_channel];
                    }
                }
            }
        }
        let active_channels = if kernel == [9, 9] {
            (0..input_channels)
                .filter(|&input_channel| {
                    (0..kernel[0] * kernel[1]).any(|spatial| {
                        let base = (spatial * input_channels + input_channel) * output_stride;
                        packed[base..base + output_channels]
                            .iter()
                            .any(|&weight| weight != 0.0)
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let weight_offset = self.packed.push(&packed)?;
        let has_bias = bias.is_some();
        let bias_offset = if let Some(mut bias) = bias {
            bias.resize(output_stride, 0.0);
            self.packed.push(&bias)?
        } else {
            0
        };
        let use_sparse_channels = !active_channels.is_empty()
            && active_channels.len() <= 16
            && active_channels.len() * 4 <= input_channels;
        let sparse_channels_offset = if use_sparse_channels {
            let encoded = active_channels
                .iter()
                .map(|&channel| f32::from_bits(channel as u32))
                .collect::<Vec<_>>();
            self.packed.push(&encoded)?
        } else {
            u32::MAX
        };
        Ok(ConvDesc {
            weight_offset,
            bias_offset,
            input_channels,
            output_channels,
            kernel,
            stride,
            padding,
            has_bias,
            depthwise: false,
            sparse_channels_offset,
            sparse_channel_count: usize::from(use_sparse_channels) * active_channels.len(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_depthwise(
        &mut self,
        prefix: &str,
        channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        has_source_bias: bool,
        norm: Option<&str>,
        layout: SourceLayout,
    ) -> Result<ConvDesc> {
        if !matches!(layout, SourceLayout::Oihw | SourceLayout::Oiw) {
            return Err(Error::InvalidModel(
                "depthwise weights must use OIHW or OIW source layout".into(),
            ));
        }
        let weight_name = format!("{prefix}.weight");
        let expected = match layout {
            SourceLayout::Oihw => vec![channels, 1, kernel[0], kernel[1]],
            SourceLayout::Oiw => vec![channels, 1, kernel[1]],
            _ => unreachable!(),
        };
        let source = read_finite(self.source, &weight_name, &expected)?;
        let bias_name = has_source_bias.then(|| format!("{prefix}.bias"));
        let (scale, bias) =
            folded_channel_params(self.source, bias_name.as_deref(), norm, channels)?;
        let channel_stride = round_up4(channels)?;
        let mut packed = vec![0.0; kernel[0] * kernel[1] * channel_stride];
        for ky in 0..kernel[0] {
            for kx in 0..kernel[1] {
                let target_base = (ky * kernel[1] + kx) * channel_stride;
                for channel in 0..channels {
                    let source_index = match layout {
                        SourceLayout::Oihw => (channel * kernel[0] + ky) * kernel[1] + kx,
                        SourceLayout::Oiw => channel * kernel[1] + kx,
                        _ => unreachable!(),
                    };
                    packed[target_base + channel] = source[source_index] * scale[channel];
                }
            }
        }
        let weight_offset = self.packed.push(&packed)?;
        let has_bias = bias.is_some();
        let bias_offset = if let Some(mut bias) = bias {
            bias.resize(channel_stride, 0.0);
            self.packed.push(&bias)?
        } else {
            0
        };
        Ok(ConvDesc {
            weight_offset,
            bias_offset,
            input_channels: channels,
            output_channels: channels,
            kernel,
            stride,
            padding,
            has_bias,
            depthwise: true,
            sparse_channels_offset: u32::MAX,
            sparse_channel_count: 0,
        })
    }
}

fn source_shape(
    layout: SourceLayout,
    input_channels: usize,
    output_channels: usize,
    kernel: [usize; 2],
) -> Result<Vec<usize>> {
    match layout {
        SourceLayout::Oihw => Ok(vec![output_channels, input_channels, kernel[0], kernel[1]]),
        SourceLayout::Oiw if kernel[0] == 1 => Ok(vec![output_channels, input_channels, kernel[1]]),
        SourceLayout::Oi if kernel == [1, 1] => Ok(vec![output_channels, input_channels]),
        SourceLayout::Iohw => Ok(vec![input_channels, output_channels, kernel[0], kernel[1]]),
        _ => Err(Error::InvalidModel(
            "source weight layout does not match the requested kernel".into(),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn source_index(
    layout: SourceLayout,
    input_channel: usize,
    output_channel: usize,
    ky: usize,
    kx: usize,
    input_channels: usize,
    output_channels: usize,
    kernel: [usize; 2],
) -> usize {
    match layout {
        SourceLayout::Oihw => {
            (((output_channel * input_channels + input_channel) * kernel[0] + ky) * kernel[1]) + kx
        }
        SourceLayout::Oiw => (output_channel * input_channels + input_channel) * kernel[1] + kx,
        SourceLayout::Oi => output_channel * input_channels + input_channel,
        SourceLayout::Iohw => {
            (((input_channel * output_channels + output_channel) * kernel[0] + ky) * kernel[1]) + kx
        }
    }
}

fn folded_channel_params(
    weights: &Weights,
    source_bias: Option<&str>,
    norm: Option<&str>,
    channels: usize,
) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    let source_bias = source_bias
        .map(|name| read_finite(weights, name, &[channels]))
        .transpose()?;
    let Some(norm) = norm else {
        return Ok((vec![1.0; channels], source_bias));
    };

    let gamma = read_finite(weights, &format!("{norm}.weight"), &[channels])?;
    let beta = read_finite(weights, &format!("{norm}.bias"), &[channels])?;
    let mean = read_finite(weights, &format!("{norm}.running_mean"), &[channels])?;
    let variance = read_finite(weights, &format!("{norm}.running_var"), &[channels])?;
    let mut scale = Vec::with_capacity(channels);
    let mut folded_bias = Vec::with_capacity(channels);
    for channel in 0..channels {
        let denominator = variance[channel] + BN_EPSILON;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(Error::InvalidModel(format!(
                "batch norm {norm:?} channel {channel} has invalid variance {}",
                variance[channel]
            )));
        }
        let channel_scale = gamma[channel] / denominator.sqrt();
        let conv_bias = source_bias.as_ref().map_or(0.0, |values| values[channel]);
        let channel_bias = beta[channel] + (conv_bias - mean[channel]) * channel_scale;
        if !channel_scale.is_finite() || !channel_bias.is_finite() {
            return Err(Error::InvalidModel(format!(
                "batch norm {norm:?} channel {channel} folds to a non-finite value"
            )));
        }
        scale.push(channel_scale);
        folded_bias.push(channel_bias);
    }
    Ok((scale, Some(folded_bias)))
}

fn read_finite(weights: &Weights, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let values = weights.tensor_with_shape(name, shape)?.to_f32()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidModel(format!(
            "tensor {name:?} contains non-finite values"
        )));
    }
    Ok(values)
}

fn round_up4(value: usize) -> Result<usize> {
    value
        .checked_add(3)
        .map(|value| value / 4 * 4)
        .ok_or_else(|| Error::InvalidModel("channel stride overflow".into()))
}

#[derive(Clone, Copy)]
struct BlockSpec {
    input_channels: usize,
    output_channels: usize,
    stride: [usize; 2],
    squeeze_excitation: bool,
}

const fn detector_neck_channels(size: ModelSize) -> usize {
    match size {
        ModelSize::Medium => 256,
        ModelSize::Small => 96,
        ModelSize::Tiny => 64,
    }
}

const fn recognizer_backbone_channels(size: ModelSize) -> usize {
    match size {
        ModelSize::Medium => 768,
        ModelSize::Small => 384,
        ModelSize::Tiny => 160,
    }
}

fn build_detector_backbone(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    size: ModelSize,
) -> Result<Vec<Value>> {
    let (mid_channels, stage_channels) = match size {
        ModelSize::Medium => (64, [128, 256, 512, 896]),
        ModelSize::Small => (24, [48, 96, 192, 384]),
        ModelSize::Tiny => (16, [32, 48, 64, 160]),
    };
    let hidden = build_large_stem(builder, input, mid_channels, stage_channels[0])?;
    build_stages(builder, hidden, &detector_stages(stage_channels))
}

fn build_recognizer_backbone(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    size: ModelSize,
) -> Result<Vec<Value>> {
    let (hidden, stages) = match size {
        ModelSize::Medium => (
            build_large_stem(builder, input, 64, 128)?,
            medium_recognizer_stages(),
        ),
        ModelSize::Small => (
            build_large_stem(builder, input, 48, 96)?,
            small_recognizer_stages(),
        ),
        ModelSize::Tiny => (build_small_stem(builder, input)?, tiny_recognizer_stages()),
    };
    build_stages(builder, hidden, &stages)
}

fn build_large_stem(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    mid_channels: usize,
    output_channels: usize,
) -> Result<Value> {
    let base = "model.backbone.encoder.convolution";
    let stem1 = builder.conv(
        input,
        &format!("{base}.stem1.convolution"),
        3,
        mid_channels,
        [3, 3],
        [2, 2],
        [1, 1],
        false,
        Some(&format!("{base}.stem1.normalization")),
        Activation::Relu,
    )?;
    let branch_hw = [stem1.shape.h, stem1.shape.w];
    let branch = builder.conv_output(
        stem1.clone(),
        &format!("{base}.stem2a.convolution"),
        mid_channels,
        mid_channels / 2,
        [2, 2],
        [1, 1],
        [0, 0],
        false,
        Some(&format!("{base}.stem2a.normalization")),
        Activation::Relu,
        branch_hw,
    )?;
    let branch = builder.conv_output(
        branch,
        &format!("{base}.stem2b.convolution"),
        mid_channels / 2,
        mid_channels,
        [2, 2],
        [1, 1],
        [0, 0],
        false,
        Some(&format!("{base}.stem2b.normalization")),
        Activation::Relu,
        branch_hw,
    )?;
    let pooled = builder
        .graph
        .max_pool(stem1, [2, 2], [1, 1], Some(branch_hw))?;
    let merged = builder.graph.concat(pooled, branch)?;
    let hidden = builder.conv(
        merged,
        &format!("{base}.stem3.convolution"),
        mid_channels * 2,
        mid_channels,
        [3, 3],
        [2, 2],
        [1, 1],
        false,
        Some(&format!("{base}.stem3.normalization")),
        Activation::Relu,
    )?;
    builder.conv(
        hidden,
        &format!("{base}.stem4.convolution"),
        mid_channels,
        output_channels,
        [1, 1],
        [1, 1],
        [0, 0],
        false,
        Some(&format!("{base}.stem4.normalization")),
        Activation::Relu,
    )
}

fn build_small_stem(builder: &mut ModelBuilder<'_>, input: Value) -> Result<Value> {
    let base = "model.backbone.encoder.convolution";
    let hidden = builder.conv(
        input,
        &format!("{base}.conv1.convolution"),
        3,
        24,
        [3, 3],
        [2, 2],
        [1, 1],
        false,
        Some(&format!("{base}.conv1.normalization")),
        Activation::Gelu,
    )?;
    builder.conv(
        hidden,
        &format!("{base}.conv2.convolution"),
        24,
        48,
        [3, 3],
        [2, 2],
        [1, 1],
        false,
        Some(&format!("{base}.conv2.normalization")),
        Activation::None,
    )
}

fn build_stages(
    builder: &mut ModelBuilder<'_>,
    mut hidden: Value,
    stages: &[Vec<BlockSpec>],
) -> Result<Vec<Value>> {
    let mut outputs = Vec::with_capacity(stages.len());
    for (stage_index, blocks) in stages.iter().enumerate() {
        for (block_index, spec) in blocks.iter().copied().enumerate() {
            hidden = build_lcnet_block(builder, hidden, stage_index, block_index, spec)?;
        }
        outputs.push(hidden.clone());
    }
    Ok(outputs)
}

fn build_lcnet_block(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    stage: usize,
    block: usize,
    spec: BlockSpec,
) -> Result<Value> {
    let base = format!("model.backbone.encoder.blocks.{stage}.blocks.{block}");
    let residual = spec.input_channels == spec.output_channels && spec.stride == [1, 1];
    let token = if residual {
        builder.depthwise(
            input,
            &format!("{base}.token_conv"),
            spec.input_channels,
            [3, 3],
            spec.stride,
            [1, 1],
            true,
            None,
            Activation::None,
            SourceLayout::Oihw,
        )?
    } else {
        builder.depthwise(
            input,
            &format!("{base}.token_conv.convolution"),
            spec.input_channels,
            [3, 3],
            spec.stride,
            [1, 1],
            false,
            Some(&format!("{base}.token_conv.normalization")),
            Activation::None,
            SourceLayout::Oihw,
        )?
    };
    let token = if spec.squeeze_excitation {
        build_squeeze_excitation(
            builder,
            token,
            &format!("{base}.token_squeeze_excitation"),
            spec.input_channels,
            Activation::HardSigmoid,
        )?
    } else {
        token
    };
    let shortcut = token.clone();
    let hidden = builder.conv(
        token,
        &format!("{base}.channel_conv1.convolution"),
        spec.input_channels,
        spec.input_channels * 2,
        [1, 1],
        [1, 1],
        [0, 0],
        false,
        Some(&format!("{base}.channel_conv1.normalization")),
        Activation::Gelu,
    )?;
    if residual {
        builder.conv_add(
            hidden,
            shortcut,
            &format!("{base}.channel_conv2.convolution"),
            spec.input_channels * 2,
            spec.output_channels,
            Some(&format!("{base}.channel_conv2.normalization")),
            Activation::None,
        )
    } else {
        builder.conv(
            hidden,
            &format!("{base}.channel_conv2.convolution"),
            spec.input_channels * 2,
            spec.output_channels,
            [1, 1],
            [1, 1],
            [0, 0],
            false,
            Some(&format!("{base}.channel_conv2.normalization")),
            Activation::None,
        )
    }
}

fn build_squeeze_excitation(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    prefix: &str,
    channels: usize,
    gate: Activation,
) -> Result<Value> {
    let pooled = builder.graph.global_mean(input.clone())?;
    let reduced = builder.conv(
        pooled,
        &format!("{prefix}.convolutions.0"),
        channels,
        channels / 4,
        [1, 1],
        [1, 1],
        [0, 0],
        true,
        None,
        Activation::Relu,
    )?;
    let attention = builder.conv(
        reduced,
        &format!("{prefix}.convolutions.2"),
        channels / 4,
        channels,
        [1, 1],
        [1, 1],
        [0, 0],
        true,
        None,
        gate,
    )?;
    builder.graph.mul_channel(input, attention)
}

fn build_variant_squeeze_excitation(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    prefix: &str,
    channels: usize,
) -> Result<Value> {
    let pooled = builder.graph.global_mean(input.clone())?;
    let reduced = builder.conv(
        pooled,
        &format!("{prefix}.conv1"),
        channels,
        channels / 4,
        [1, 1],
        [1, 1],
        [0, 0],
        true,
        None,
        Activation::Relu,
    )?;
    let attention = builder.conv(
        reduced,
        &format!("{prefix}.conv2"),
        channels / 4,
        channels,
        [1, 1],
        [1, 1],
        [0, 0],
        true,
        None,
        Activation::HardSigmoidFive,
    )?;
    builder.graph.mul_channel(input, attention)
}

fn build_detector_neck(
    builder: &mut ModelBuilder<'_>,
    stages: &[Value],
    size: ModelSize,
) -> Result<Value> {
    match size {
        ModelSize::Medium => build_medium_detector_neck(builder, stages),
        ModelSize::Small => build_replk_detector_neck(builder, stages, [48, 96, 192, 384], 96, 7),
        ModelSize::Tiny => build_replk_detector_neck(builder, stages, [32, 48, 64, 160], 64, 5),
    }
}

fn build_replk_detector_neck(
    builder: &mut ModelBuilder<'_>,
    stages: &[Value],
    stage_channels: [usize; 4],
    neck_channels: usize,
    kernel_size: usize,
) -> Result<Value> {
    if stages.len() != 4 {
        return Err(Error::InvalidModel(format!(
            "detector backbone returned {} stages; expected 4",
            stages.len()
        )));
    }
    let mut fused = Vec::with_capacity(4);
    for (index, (stage, input_channels)) in stages.iter().cloned().zip(stage_channels).enumerate() {
        let prefix = format!("model.neck.insert_conv.{index}");
        let hidden = builder.conv(
            stage,
            &format!("{prefix}.in_conv"),
            input_channels,
            neck_channels,
            [1, 1],
            [1, 1],
            [0, 0],
            false,
            None,
            Activation::None,
        )?;
        let attention = build_variant_squeeze_excitation(
            builder,
            hidden.clone(),
            &format!("{prefix}.squeeze_excitation_block"),
            neck_channels,
        )?;
        fused.push(builder.graph.add(hidden, attention)?);
    }
    for index in (0..3).rev() {
        let output_hw = [fused[index].shape.h, fused[index].shape.w];
        let upper = builder
            .graph
            .resize_nearest(fused[index + 1].clone(), output_hw)?;
        fused[index] = builder.graph.add(fused[index].clone(), upper)?;
    }

    let mut features = Vec::with_capacity(4);
    for (index, feature) in fused.into_iter().enumerate() {
        let prefix = format!("model.neck.input_conv.{index}");
        let hidden = builder.depthwise(
            feature,
            &format!("{prefix}.depthwise_convolution"),
            neck_channels,
            [kernel_size, kernel_size],
            [1, 1],
            [kernel_size / 2, kernel_size / 2],
            true,
            None,
            Activation::None,
            SourceLayout::Oihw,
        )?;
        let hidden = builder.conv(
            hidden,
            &format!("{prefix}.pointwise_convolution"),
            neck_channels,
            neck_channels / 4,
            [1, 1],
            [1, 1],
            [0, 0],
            false,
            None,
            Activation::None,
        )?;
        let attention = build_variant_squeeze_excitation(
            builder,
            hidden.clone(),
            &format!("{prefix}.squeeze_excitation_module"),
            neck_channels / 4,
        )?;
        features.push(builder.graph.add(hidden, attention)?);
    }

    let output_hw = [features[0].shape.h, features[0].shape.w];
    for (index, feature) in features.iter_mut().enumerate().skip(1) {
        let expected = [
            feature.shape.h * (1 << index),
            feature.shape.w * (1 << index),
        ];
        if expected != output_hw {
            return Err(Error::InvalidModel(format!(
                "detector FPN scale {index} produces {expected:?}; expected {output_hw:?}"
            )));
        }
        *feature = builder.graph.resize_nearest(feature.clone(), output_hw)?;
    }
    features.reverse();
    let mut output = features.remove(0);
    for feature in features {
        output = builder.graph.concat(output, feature)?;
    }
    Ok(output)
}

fn build_medium_detector_neck(builder: &mut ModelBuilder<'_>, stages: &[Value]) -> Result<Value> {
    if stages.len() != 4 {
        return Err(Error::InvalidModel(format!(
            "detector backbone returned {} stages; expected 4",
            stages.len()
        )));
    }
    let stage_channels = [128, 256, 512, 896];
    let mut adjusted = Vec::with_capacity(4);
    for (index, (stage, input_channels)) in stages.iter().cloned().zip(stage_channels).enumerate() {
        adjusted.push(builder.conv(
            stage,
            &format!("model.neck.input_channel_adjustment_convolution.{index}"),
            input_channels,
            256,
            [1, 1],
            [1, 1],
            [0, 0],
            false,
            None,
            Activation::None,
        )?);
    }

    let mut top_down = adjusted.clone();
    for index in (0..3).rev() {
        let output_hw = [top_down[index].shape.h, top_down[index].shape.w];
        let upper = builder
            .graph
            .resize_nearest(top_down[index + 1].clone(), output_hw)?;
        top_down[index] = builder.graph.add(top_down[index].clone(), upper)?;
    }

    let mut projected = Vec::with_capacity(4);
    for (index, source) in top_down.into_iter().enumerate() {
        projected.push(builder.conv(
            source,
            &format!("model.neck.input_feature_projection_convolution.{index}"),
            256,
            64,
            [9, 9],
            [1, 1],
            [4, 4],
            true,
            None,
            Activation::None,
        )?);
    }

    let mut bottom_up = vec![projected[0].clone()];
    for index in 1..4 {
        let lower = builder.conv(
            bottom_up[index - 1].clone(),
            &format!("model.neck.path_aggregation_head_convolution.{}", index - 1),
            64,
            64,
            [3, 3],
            [2, 2],
            [1, 1],
            false,
            None,
            Activation::None,
        )?;
        bottom_up.push(builder.graph.add(projected[index].clone(), lower)?);
    }

    let mut refined = Vec::with_capacity(4);
    for (index, source) in bottom_up.into_iter().enumerate() {
        let lateral = builder.conv(
            source,
            &format!("model.neck.path_aggregation_lateral_convolution.{index}"),
            64,
            64,
            [9, 9],
            [1, 1],
            [4, 4],
            true,
            None,
            Activation::None,
        )?;
        refined.push(build_intraclass_block(builder, lateral, index)?);
    }

    let output_hw = [refined[0].shape.h, refined[0].shape.w];
    for (index, feature) in refined.iter_mut().enumerate().skip(1) {
        let expected = [
            feature.shape.h * (1 << index),
            feature.shape.w * (1 << index),
        ];
        if expected != output_hw {
            return Err(Error::InvalidModel(format!(
                "medium detector FPN scale {index} produces {expected:?}; expected {output_hw:?}"
            )));
        }
        *feature = builder.graph.resize_nearest(feature.clone(), output_hw)?;
    }
    refined.reverse();
    let mut output = refined.remove(0);
    for feature in refined {
        output = builder.graph.concat(output, feature)?;
    }
    Ok(output)
}

fn build_intraclass_block(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    index: usize,
) -> Result<Value> {
    let prefix = format!("model.neck.intraclass_blocks.{index}");
    let reduced = builder.conv(
        input.clone(),
        &format!("{prefix}.conv_reduce_channel"),
        64,
        32,
        [1, 1],
        [1, 1],
        [0, 0],
        true,
        None,
        Activation::None,
    )?;
    let layer7 = build_intraclass_layer(builder, reduced, &prefix, 7, "longratio")?;
    let layer5 = build_intraclass_layer(builder, layer7, &prefix, 5, "midratio")?;
    let layer3 = build_intraclass_layer(builder, layer5, &prefix, 3, "shortratio")?;
    let final_conv = builder.conv(
        layer3,
        &format!("{prefix}.conv_final.convolution"),
        32,
        64,
        [1, 1],
        [1, 1],
        [0, 0],
        true,
        Some(&format!("{prefix}.conv_final.norm")),
        Activation::Relu,
    )?;
    builder.graph.add(input, final_conv)
}

fn build_intraclass_layer(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    prefix: &str,
    kernel: usize,
    ratio: &str,
) -> Result<Value> {
    let symmetric = builder.conv(
        input.clone(),
        &format!("{prefix}.symmetric_conv_long_{ratio}"),
        32,
        32,
        [kernel, kernel],
        [1, 1],
        [kernel / 2, kernel / 2],
        true,
        None,
        Activation::None,
    )?;
    let vertical = builder.conv(
        input.clone(),
        &format!("{prefix}.vertical_long_to_small_conv_{ratio}"),
        32,
        32,
        [kernel, 1],
        [1, 1],
        [kernel / 2, 0],
        true,
        None,
        Activation::None,
    )?;
    let horizontal = builder.conv(
        input,
        &format!("{prefix}.horizontal_small_to_long_conv_{ratio}"),
        32,
        32,
        [1, kernel],
        [1, 1],
        [0, kernel / 2],
        true,
        None,
        Activation::None,
    )?;
    let merged = builder.graph.add(symmetric, vertical)?;
    builder.graph.add(merged, horizontal)
}

fn build_detector_head(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    input_channels: usize,
) -> Result<Value> {
    let hidden_channels = input_channels / 4;
    if input_channels <= 96 {
        return builder.detector_head(input, input_channels, hidden_channels);
    }
    let hidden = builder.conv(
        input,
        "head.conv_down.convolution",
        input_channels,
        hidden_channels,
        [3, 3],
        [1, 1],
        [1, 1],
        false,
        Some("head.conv_down.norm"),
        Activation::Relu,
    )?;
    let hidden = builder.deconv(
        hidden,
        "head.conv_up.convolution",
        hidden_channels,
        hidden_channels,
        Some("head.conv_up.norm"),
        Activation::Relu,
    )?;
    builder.deconv(
        hidden,
        "head.conv_final",
        hidden_channels,
        1,
        None,
        Activation::Sigmoid,
    )
}

fn build_recognizer_head(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    size: ModelSize,
) -> Result<Value> {
    match size {
        ModelSize::Medium | ModelSize::Small => {
            build_light_svtr_recognizer_head(builder, input, size)
        }
        ModelSize::Tiny => build_tiny_recognizer_head(builder, input),
    }
}

fn build_tiny_recognizer_head(builder: &mut ModelBuilder<'_>, input: Value) -> Result<Value> {
    let hidden = builder.depthwise(
        input,
        "head.conv1",
        160,
        [1, 5],
        [1, 1],
        [0, 2],
        false,
        Some("head.norm1"),
        Activation::HardSwish,
        SourceLayout::Oiw,
    )?;
    let hidden = builder.conv_1d(
        hidden,
        "head.conv2",
        160,
        160,
        1,
        0,
        false,
        Some("head.norm2"),
        Activation::HardSwish,
    )?;
    let hidden = builder.linear(hidden, "head.fc1", 160, 80, Activation::None)?;
    let logits = builder.linear(
        hidden,
        "head.fc2",
        80,
        TINY_RECOGNIZER_CLASSES,
        Activation::None,
    )?;
    builder.graph.softmax(logits)
}

fn build_light_svtr_recognizer_head(
    builder: &mut ModelBuilder<'_>,
    input: Value,
    size: ModelSize,
) -> Result<Value> {
    let (input_channels, hidden_channels, mlp_channels) = match size {
        ModelSize::Medium => (768, 192, 768),
        ModelSize::Small => (384, 120, 240),
        ModelSize::Tiny => {
            return Err(Error::InvalidModel(
                "tiny recognizer does not use the LightSVTR head".into(),
            ));
        }
    };
    let residual = builder.conv(
        input.clone(),
        "head.encoder.conv_block.0.convolution",
        input_channels,
        hidden_channels,
        [1, 1],
        [1, 1],
        [0, 0],
        false,
        Some("head.encoder.conv_block.0.normalization"),
        Activation::Silu,
    )?;
    let hidden = builder.conv(
        input,
        "head.encoder.conv_block.1.convolution",
        input_channels,
        hidden_channels,
        [1, 1],
        [1, 1],
        [0, 0],
        false,
        Some("head.encoder.conv_block.1.normalization"),
        Activation::Silu,
    )?;
    let local = builder.depthwise(
        hidden.clone(),
        "head.encoder.conv_block.2.convolution",
        hidden_channels,
        [1, 7],
        [1, 1],
        [0, 3],
        false,
        Some("head.encoder.conv_block.2.normalization"),
        Activation::Silu,
        SourceLayout::Oihw,
    )?;
    let mut hidden = builder.graph.add(hidden, local)?;

    for index in 0..2 {
        let prefix = format!("head.encoder.svtr_block.{index}");
        let normalized = builder.layer_norm(
            hidden.clone(),
            &format!("{prefix}.layer_norm1"),
            hidden_channels,
        )?;
        let qkv = builder.linear(
            normalized,
            &format!("{prefix}.self_attn.qkv"),
            hidden_channels,
            hidden_channels * 3,
            Activation::None,
        )?;
        let attended = builder.graph.attention(qkv, hidden_channels, 8)?;
        let projected = builder.linear(
            attended,
            &format!("{prefix}.self_attn.projection"),
            hidden_channels,
            hidden_channels,
            Activation::None,
        )?;
        hidden = builder.graph.add(hidden, projected)?;

        let normalized = builder.layer_norm(
            hidden.clone(),
            &format!("{prefix}.layer_norm2"),
            hidden_channels,
        )?;
        let mlp = builder.linear(
            normalized,
            &format!("{prefix}.mlp.fc1"),
            hidden_channels,
            mlp_channels,
            Activation::Silu,
        )?;
        let mlp = builder.linear(
            mlp,
            &format!("{prefix}.mlp.fc2"),
            mlp_channels,
            hidden_channels,
            Activation::None,
        )?;
        hidden = builder.graph.add(hidden, mlp)?;
    }

    let hidden = builder.layer_norm(hidden, "head.encoder.norm", hidden_channels)?;
    let hidden = builder.graph.add(hidden, residual)?;
    let logits = builder.linear(
        hidden,
        "head.head",
        hidden_channels,
        LARGE_RECOGNIZER_CLASSES,
        Activation::None,
    )?;
    builder.graph.softmax(logits)
}

fn block(
    input_channels: usize,
    output_channels: usize,
    stride: [usize; 2],
    squeeze_excitation: bool,
) -> BlockSpec {
    BlockSpec {
        input_channels,
        output_channels,
        stride,
        squeeze_excitation,
    }
}

fn detector_stages(channels: [usize; 4]) -> Vec<Vec<BlockSpec>> {
    let [c1, c2, c3, c4] = channels;
    vec![
        vec![block(c1, c1, [1, 1], true), block(c1, c1, [1, 1], false)],
        vec![
            block(c1, c2, [2, 2], false),
            block(c2, c2, [1, 1], true),
            block(c2, c2, [1, 1], false),
        ],
        vec![
            block(c2, c3, [2, 2], false),
            block(c3, c3, [1, 1], true),
            block(c3, c3, [1, 1], false),
            block(c3, c3, [1, 1], true),
            block(c3, c3, [1, 1], false),
        ],
        vec![
            block(c3, c4, [2, 2], false),
            block(c4, c4, [1, 1], true),
            block(c4, c4, [1, 1], false),
        ],
    ]
}

fn medium_recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![block(128, 128, [1, 1], true)],
        vec![
            block(128, 256, [1, 1], false),
            block(256, 256, [1, 1], false),
            block(256, 256, [1, 1], true),
        ],
        vec![
            block(256, 512, [2, 1], false),
            block(512, 512, [1, 1], true),
            block(512, 512, [1, 1], false),
            block(512, 512, [1, 1], true),
            block(512, 512, [1, 1], false),
            block(512, 512, [1, 1], true),
            block(512, 512, [1, 1], false),
        ],
        vec![
            block(512, 768, [2, 1], false),
            block(768, 768, [1, 1], true),
            block(768, 768, [1, 1], false),
        ],
    ]
}

fn small_recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![block(96, 96, [1, 1], true)],
        vec![block(96, 96, [1, 1], false), block(96, 96, [1, 1], false)],
        vec![
            block(96, 192, [2, 1], false),
            block(192, 192, [1, 1], true),
            block(192, 192, [1, 1], false),
            block(192, 192, [1, 1], true),
            block(192, 192, [1, 1], false),
            block(192, 192, [1, 1], true),
            block(192, 192, [1, 1], false),
        ],
        vec![
            block(192, 384, [2, 1], false),
            block(384, 384, [1, 1], true),
            block(384, 384, [1, 1], false),
        ],
    ]
}

fn tiny_recognizer_stages() -> Vec<Vec<BlockSpec>> {
    vec![
        vec![block(48, 48, [1, 1], true)],
        vec![block(48, 48, [1, 1], false)],
        vec![
            block(48, 96, [2, 1], false),
            block(96, 96, [1, 1], true),
            block(96, 96, [1, 1], false),
        ],
        vec![
            block(96, 160, [2, 1], false),
            block(160, 160, [1, 1], true),
            block(160, 160, [1, 1], false),
            block(160, 160, [1, 1], false),
        ],
    ]
}
