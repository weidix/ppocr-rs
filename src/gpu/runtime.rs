use super::error::{Error, Result};
use std::{
    cell::RefCell,
    future::Future,
    pin::pin,
    rc::Rc,
    sync::{Arc, Mutex, mpsc},
    task::{Context, Poll, Wake, Waker},
    thread,
};

const IMMEDIATE_WORDS: usize = 32;
const IMMEDIATE_BYTES: u32 = (IMMEDIATE_WORDS * size_of::<u32>()) as u32;
const INVALID_OFFSET: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
}

#[derive(Clone)]
pub struct Gpu {
    inner: Arc<GpuInner>,
}

struct GpuInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Kernels,
    info: GpuInfo,
    timestamp_profiling: bool,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let backends = platform_backends();
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|error| Error::Gpu(format!("request {backends:?} adapter: {error}")))?;

        let adapter_info = adapter.get_info();
        let expected = platform_backend();
        if adapter_info.backend != expected {
            return Err(Error::Gpu(format!(
                "selected backend {:?}; expected {expected:?}",
                adapter_info.backend
            )));
        }
        let adapter_features = adapter.features();
        if !adapter_features.contains(wgpu::Features::IMMEDIATES) {
            return Err(Error::Gpu(
                "the selected adapter does not support wgpu immediates".into(),
            ));
        }
        let adapter_limits = adapter.limits();
        if adapter_limits.max_immediate_size < IMMEDIATE_BYTES {
            return Err(Error::Gpu(format!(
                "adapter immediate limit is {} bytes; {IMMEDIATE_BYTES} required",
                adapter_limits.max_immediate_size
            )));
        }

        let mut required_features = wgpu::Features::IMMEDIATES;
        let timestamp_profiling = adapter_features.contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        );
        if timestamp_profiling {
            required_features |=
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ppocr-gpu"),
            required_features,
            required_limits: adapter_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|error| Error::Gpu(format!("create device: {error}")))?;
        let kernels = Kernels::new(&device)?;
        Ok(Self {
            inner: Arc::new(GpuInner {
                device,
                queue,
                kernels,
                info: GpuInfo {
                    name: adapter_info.name,
                    backend: adapter_info.backend,
                    device_type: adapter_info.device_type,
                },
                timestamp_profiling,
            }),
        })
    }

    pub fn info(&self) -> &GpuInfo {
        &self.inner.info
    }

    pub(crate) fn create_session(&self, weights: Vec<f32>, plan: Plan) -> Result<Session> {
        Session::new(self.clone(), weights, plan)
    }
}

#[cfg(target_os = "macos")]
const fn platform_backends() -> wgpu::Backends {
    wgpu::Backends::METAL
}

#[cfg(not(target_os = "macos"))]
const fn platform_backends() -> wgpu::Backends {
    wgpu::Backends::VULKAN
}

#[cfg(target_os = "macos")]
const fn platform_backend() -> wgpu::Backend {
    wgpu::Backend::Metal
}

#[cfg(not(target_os = "macos"))]
const fn platform_backend() -> wgpu::Backend {
    wgpu::Backend::Vulkan
}

struct ThreadWake(thread::Thread);

impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Activation {
    None = 0,
    Relu = 1,
    Silu = 2,
    HardSigmoid = 3,
    HardSigmoidFive = 4,
    Gelu = 5,
    HardSwish = 6,
    Sigmoid = 7,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvDesc {
    pub weight_offset: u32,
    pub bias_offset: u32,
    pub input_channels: usize,
    pub output_channels: usize,
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub has_bias: bool,
    pub depthwise: bool,
    pub sparse_channels_offset: u32,
    pub sparse_channel_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Shape4 {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub cs: usize,
}

impl Shape4 {
    fn new(n: usize, h: usize, w: usize, c: usize) -> Result<Self> {
        if n == 0 || h == 0 || w == 0 || c == 0 {
            return Err(Error::InvalidInput(format!(
                "tensor dimensions must be nonzero, got [{n}, {h}, {w}, {c}]"
            )));
        }
        let cs = c
            .checked_add(3)
            .map(|value| value / 4 * 4)
            .ok_or_else(|| Error::InvalidInput("channel stride overflow".into()))?;
        let shape = Self { n, h, w, c, cs };
        shape.elements()?;
        Ok(shape)
    }

    fn elements(self) -> Result<usize> {
        [self.n, self.h, self.w, self.cs]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .ok_or_else(|| Error::InvalidInput(format!("tensor shape {self:?} overflows")))
    }

    fn logical_elements(self) -> Result<usize> {
        [self.n, self.h, self.w, self.c]
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .ok_or_else(|| Error::InvalidInput(format!("tensor shape {self:?} overflows")))
    }
}

#[derive(Clone)]
pub(crate) struct Value {
    allocation: Rc<Allocation>,
    pub shape: Shape4,
}

impl Value {
    fn offset(&self) -> u32 {
        self.allocation.offset
    }
}

struct Allocation {
    offset: u32,
    length: u32,
    allocator: Rc<RefCell<ArenaAllocator>>,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.allocator
            .borrow_mut()
            .release(self.offset, self.length);
    }
}

#[derive(Clone, Copy, Debug)]
struct Region {
    offset: u32,
    length: u32,
}

#[derive(Default)]
struct ArenaAllocator {
    end: u32,
    high_water: u32,
    free: Vec<Region>,
}

impl ArenaAllocator {
    fn allocate(&mut self, length: usize) -> Result<(u32, u32)> {
        let length = align4(length)?;
        if let Some((index, region)) = self
            .free
            .iter()
            .copied()
            .enumerate()
            .find(|(_, region)| region.length >= length)
        {
            self.free.remove(index);
            if region.length > length {
                self.free.push(Region {
                    offset: region.offset + length,
                    length: region.length - length,
                });
            }
            return Ok((region.offset, length));
        }
        let offset = self.end;
        self.end = self
            .end
            .checked_add(length)
            .ok_or_else(|| Error::Gpu("activation arena exceeds u32 indexing".into()))?;
        self.high_water = self.high_water.max(self.end);
        Ok((offset, length))
    }

    fn release(&mut self, offset: u32, length: u32) {
        self.free.push(Region { offset, length });
        self.free.sort_unstable_by_key(|region| region.offset);
        let mut merged: Vec<Region> = Vec::with_capacity(self.free.len());
        for region in self.free.drain(..) {
            if let Some(last) = merged.last_mut()
                && last.offset + last.length == region.offset
            {
                last.length += region.length;
                continue;
            }
            merged.push(region);
        }
        self.free = merged;
    }
}

fn align4(length: usize) -> Result<u32> {
    let length = length
        .checked_add(3)
        .map(|value| value / 4 * 4)
        .ok_or_else(|| Error::Gpu("buffer length overflow".into()))?;
    u32::try_from(length).map_err(|_| Error::Gpu("buffer exceeds u32 indexing".into()))
}

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Conv,
    Conv2x2Direct,
    Conv3x3Direct,
    Conv3x3Stride2Direct,
    ConvLarge,
    ConvSparse9,
    ConvSpatialM32,
    ConvMediumLinear,
    ConvSingleRow,
    Depthwise,
    Add,
    MulChannel,
    ResizeNearest,
    Concat,
    GlobalMean,
    PoolMax,
    PoolAvg,
    Deconv,
    DeconvFinal,
    DeconvPhase,
    FusedDetectorHead,
    Softmax,
    LayerNorm,
    AttentionScores,
    AttentionContext,
}

#[derive(Clone)]
struct Dispatch {
    kernel: Kernel,
    params: [u32; IMMEDIATE_WORDS],
    workgroups: [u32; 3],
}

pub(crate) struct Plan {
    input_offset: u32,
    input_shape: Shape4,
    output_offset: u32,
    output_shape: Shape4,
    arena_elements: u32,
    dispatches: Vec<Dispatch>,
}

pub(crate) struct GraphBuilder {
    allocator: Rc<RefCell<ArenaAllocator>>,
    _input: Value,
    input_offset: u32,
    input_shape: Shape4,
    dispatches: Vec<Dispatch>,
}

impl GraphBuilder {
    pub fn new(nchw: [usize; 4]) -> Result<(Self, Value)> {
        let [n, c, h, w] = nchw;
        let input_shape = Shape4::new(n, h, w, c)?;
        let allocator = Rc::new(RefCell::new(ArenaAllocator::default()));
        let input = allocate_value(&allocator, input_shape)?;
        let input_offset = input.offset();
        Ok((
            Self {
                allocator,
                _input: input.clone(),
                input_offset,
                input_shape,
                dispatches: Vec::new(),
            },
            input,
        ))
    }

    pub fn conv(&mut self, input: Value, desc: &ConvDesc, activation: Activation) -> Result<Value> {
        self.conv_impl(input, desc, activation, None, None)
    }

    pub fn conv_output(
        &mut self,
        input: Value,
        desc: &ConvDesc,
        activation: Activation,
        output_hw: [usize; 2],
    ) -> Result<Value> {
        self.conv_impl(input, desc, activation, None, Some(output_hw))
    }

    pub fn depthwise(
        &mut self,
        input: Value,
        desc: &ConvDesc,
        activation: Activation,
    ) -> Result<Value> {
        if !desc.depthwise {
            return Err(Error::InvalidModel(
                "depthwise dispatch received an ungrouped convolution".into(),
            ));
        }
        self.conv(input, desc, activation)
    }

    pub fn conv_add(
        &mut self,
        input: Value,
        desc: &ConvDesc,
        activation: Activation,
        add: Value,
    ) -> Result<Value> {
        self.conv_impl(input, desc, activation, Some(add), None)
    }

    fn conv_impl(
        &mut self,
        input: Value,
        desc: &ConvDesc,
        activation: Activation,
        add: Option<Value>,
        output_hw: Option<[usize; 2]>,
    ) -> Result<Value> {
        if input.shape.c != desc.input_channels {
            return Err(Error::InvalidModel(format!(
                "convolution expects {} channels, found {}",
                desc.input_channels, input.shape.c
            )));
        }
        let [output_h, output_w] = output_hw.unwrap_or([
            conv_output_dim(
                input.shape.h,
                desc.kernel[0],
                desc.stride[0],
                desc.padding[0],
            )?,
            conv_output_dim(
                input.shape.w,
                desc.kernel[1],
                desc.stride[1],
                desc.padding[1],
            )?,
        ]);
        let output_shape = Shape4::new(input.shape.n, output_h, output_w, desc.output_channels)?;
        if let Some(add) = &add
            && add.shape != output_shape
        {
            return Err(Error::InvalidModel(format!(
                "fused convolution add shape {:?} does not match output {output_shape:?}",
                add.shape
            )));
        }
        let output = allocate_value(&self.allocator, output_shape)?;
        let mut params = [0u32; IMMEDIATE_WORDS];
        params[0] = input.offset();
        params[1] = output.offset();
        params[2] = add.as_ref().map_or(INVALID_OFFSET, Value::offset);
        params[3] = desc.weight_offset;
        params[4] = desc.bias_offset;
        params[5] = to_u32(input.shape.n, "batch")?;
        params[6] = to_u32(input.shape.h, "input height")?;
        params[7] = to_u32(input.shape.w, "input width")?;
        params[8] = to_u32(input.shape.c, "input channels")?;
        params[9] = to_u32(input.shape.cs, "input channel stride")?;
        params[10] = to_u32(output_shape.h, "output height")?;
        params[11] = to_u32(output_shape.w, "output width")?;
        params[12] = to_u32(output_shape.c, "output channels")?;
        params[13] = to_u32(output_shape.cs, "output channel stride")?;
        params[14] = to_u32(desc.kernel[0], "kernel height")?;
        params[15] = to_u32(desc.kernel[1], "kernel width")?;
        params[16] = to_u32(desc.stride[0], "stride y")?;
        params[17] = to_u32(desc.stride[1], "stride x")?;
        params[18] = 1;
        params[19] = 1;
        params[20] = to_u32(desc.padding[0], "padding y")?;
        params[21] = to_u32(desc.padding[1], "padding x")?;
        params[22] = to_u32(output_shape.cs, "weight K stride")?;
        params[23] = activation as u32;
        params[24] = u32::from(desc.has_bias) | (u32::from(add.is_some()) << 1);
        params[27] = desc.sparse_channels_offset;
        params[28] = to_u32(desc.sparse_channel_count, "sparse channel count")?;
        let rows = input
            .shape
            .n
            .checked_mul(output_shape.h)
            .and_then(|value| value.checked_mul(output_shape.w))
            .ok_or_else(|| Error::InvalidModel("convolution output rows overflow".into()))?;
        let (kernel, rows_per_workgroup, channels_per_workgroup) = if desc.depthwise {
            (Kernel::Depthwise, 8, 32)
        } else if desc.kernel == [9, 9]
            && desc.stride == [1, 1]
            && desc.padding == [4, 4]
            && output_shape.c == 64
            && input.shape.c.is_multiple_of(32)
            && desc.sparse_channel_count != 0
        {
            (Kernel::ConvSparse9, 32, 64)
        } else if desc.kernel == [9, 9]
            && desc.stride == [1, 1]
            && desc.padding == [4, 4]
            && output_shape.c == 64
            && input.shape.c.is_multiple_of(32)
        {
            (Kernel::ConvLarge, 32, 64)
        } else if input.shape.n == 1
            && input.shape.h == 208
            && input.shape.w == 368
            && matches!([input.shape.c, output_shape.c], [64, 32] | [32, 64])
            && output_shape.h == 208
            && output_shape.w == 368
            && desc.kernel == [2, 2]
            && desc.stride == [1, 1]
            && desc.padding == [0, 0]
            && activation == Activation::Relu
            && add.is_none()
        {
            (Kernel::Conv2x2Direct, 64, 32)
        } else if input.shape.n == 1
            && input.shape.h == 208
            && input.shape.w == 368
            && input.shape.c == 128
            && output_shape.h == 104
            && output_shape.w == 184
            && output_shape.c == 64
            && desc.kernel == [3, 3]
            && desc.stride == [2, 2]
            && desc.padding == [1, 1]
            && activation == Activation::Relu
            && add.is_none()
        {
            (Kernel::Conv3x3Stride2Direct, 64, 32)
        } else if input.shape.n == 1
            && input.shape.h == 104
            && input.shape.w == 184
            && input.shape.c == 256
            && output_shape.h == 104
            && output_shape.w == 184
            && output_shape.c == 64
            && desc.kernel == [3, 3]
            && desc.stride == [1, 1]
            && desc.padding == [1, 1]
            && activation == Activation::Relu
            && add.is_none()
        {
            (Kernel::Conv3x3Direct, 64, 32)
        } else if input.shape.n == 1
            && input.shape.c == 32
            && output_shape.c == 32
            && input.shape.h == output_shape.h
            && input.shape.w == output_shape.w
            && matches!(
                [output_shape.h, output_shape.w],
                [104, 184] | [52, 92] | [26, 46] | [13, 23]
            )
            && matches!(
                desc.kernel,
                [7, 7] | [7, 1] | [1, 7] | [5, 5] | [5, 1] | [1, 5] | [3, 3] | [3, 1] | [1, 3]
            )
            && desc.stride == [1, 1]
            && desc.padding == [desc.kernel[0] / 2, desc.kernel[1] / 2]
            && activation == Activation::None
            && add.is_none()
        {
            (Kernel::ConvSpatialM32, 32, 32)
        } else if desc.kernel == [1, 1]
            && desc.stride == [1, 1]
            && desc.padding == [0, 0]
            && rows == 1
            && input.shape.c.is_multiple_of(4)
            && input.shape.c <= 768
            && output_shape.c >= 32
        {
            (Kernel::ConvSingleRow, 1, 32)
        } else if desc.kernel == [1, 1]
            && desc.stride == [1, 1]
            && desc.padding == [0, 0]
            && rows >= 32
            && output_shape.c >= 64
            && input.shape.c.is_multiple_of(4)
        {
            (Kernel::ConvMediumLinear, 64, 64)
        } else if output_shape.c > 1_024 {
            (Kernel::Conv, 16, 64)
        } else if desc.kernel == [1, 1] && rows >= 64 {
            (Kernel::Conv, 32, 32)
        } else {
            (Kernel::Conv, 8, 32)
        };
        let row_workgroups = match kernel {
            Kernel::ConvLarge | Kernel::ConvSparse9 | Kernel::ConvSpatialM32 => {
                let tiles_per_row = output_shape.w.div_ceil(32);
                params[26] = to_u32(tiles_per_row, "spatial convolution tiles per row")?;
                let groups = output_shape
                    .n
                    .checked_mul(output_shape.h)
                    .and_then(|value| value.checked_mul(tiles_per_row))
                    .ok_or_else(|| {
                        Error::InvalidModel("spatial convolution dispatch size overflow".into())
                    })?;
                to_u32(groups, "large convolution row workgroups")?
            }
            Kernel::Conv2x2Direct | Kernel::Conv3x3Direct | Kernel::Conv3x3Stride2Direct => {
                let tiles_per_row = output_shape.w.div_ceil(8);
                let tile_rows = output_shape.h.div_ceil(8);
                params[26] = to_u32(tiles_per_row, "direct convolution tiles per row")?;
                let groups = output_shape
                    .n
                    .checked_mul(tile_rows)
                    .and_then(|value| value.checked_mul(tiles_per_row))
                    .ok_or_else(|| {
                        Error::InvalidModel("direct convolution dispatch size overflow".into())
                    })?;
                to_u32(groups, "direct convolution workgroups")?
            }
            _ => div_ceil_u32(rows, rows_per_workgroup)?,
        };
        self.dispatches.push(Dispatch {
            kernel,
            params,
            workgroups: [
                row_workgroups,
                div_ceil_u32(output_shape.c, channels_per_workgroup)?,
                1,
            ],
        });
        Ok(output)
    }

    pub fn add(&mut self, left: Value, right: Value) -> Result<Value> {
        if left.shape != right.shape {
            return Err(Error::InvalidModel(format!(
                "add shapes differ: {:?} and {:?}",
                left.shape, right.shape
            )));
        }
        self.elementwise_binary(left, right, Kernel::Add, Activation::None)
    }

    pub fn mul_channel(&mut self, input: Value, channel: Value) -> Result<Value> {
        if channel.shape.c != input.shape.c
            || (channel.shape.n != 1 && channel.shape.n != input.shape.n)
            || channel.shape.h != 1
            || channel.shape.w != 1
        {
            return Err(Error::InvalidModel(format!(
                "channel broadcast shape {:?} is incompatible with {:?}",
                channel.shape, input.shape
            )));
        }
        self.elementwise_binary(input, channel, Kernel::MulChannel, Activation::None)
    }

    fn elementwise_binary(
        &mut self,
        left: Value,
        right: Value,
        kernel: Kernel,
        activation: Activation,
    ) -> Result<Value> {
        let output_shape = left.shape;
        let output = allocate_value(&self.allocator, output_shape)?;
        let params = elementwise_params(&left, Some(&right), &output, activation, output_shape)?;
        self.dispatches.push(Dispatch {
            kernel,
            params,
            workgroups: [elementwise_workgroups(output_shape)?, 1, 1],
        });
        Ok(output)
    }

    pub fn resize_nearest(&mut self, input: Value, output_hw: [usize; 2]) -> Result<Value> {
        let output_shape = Shape4::new(input.shape.n, output_hw[0], output_hw[1], input.shape.c)?;
        let output = allocate_value(&self.allocator, output_shape)?;
        let params = elementwise_params(&input, None, &output, Activation::None, output_shape)?;
        self.dispatches.push(Dispatch {
            kernel: Kernel::ResizeNearest,
            params,
            workgroups: [elementwise_workgroups(output_shape)?, 1, 1],
        });
        Ok(output)
    }

    pub fn concat(&mut self, left: Value, right: Value) -> Result<Value> {
        if (left.shape.n, left.shape.h, left.shape.w)
            != (right.shape.n, right.shape.h, right.shape.w)
        {
            return Err(Error::InvalidModel(format!(
                "concat spatial shapes differ: {:?} and {:?}",
                left.shape, right.shape
            )));
        }
        let channels = left
            .shape
            .c
            .checked_add(right.shape.c)
            .ok_or_else(|| Error::InvalidModel("concat channels overflow".into()))?;
        let output_shape = Shape4::new(left.shape.n, left.shape.h, left.shape.w, channels)?;
        let output = allocate_value(&self.allocator, output_shape)?;
        let params =
            elementwise_params(&left, Some(&right), &output, Activation::None, output_shape)?;
        self.dispatches.push(Dispatch {
            kernel: Kernel::Concat,
            params,
            workgroups: [elementwise_workgroups(output_shape)?, 1, 1],
        });
        Ok(output)
    }

    pub fn global_mean(&mut self, input: Value) -> Result<Value> {
        let output_shape = Shape4::new(input.shape.n, 1, 1, input.shape.c)?;
        let output = allocate_value(&self.allocator, output_shape)?;
        let params = pool_params(
            &input,
            &output,
            [input.shape.h, input.shape.w],
            [1, 1],
            [0, 0],
            Activation::None,
            false,
        )?;
        self.dispatches.push(Dispatch {
            kernel: Kernel::GlobalMean,
            params,
            workgroups: [
                div_ceil_u32(output_shape.cs, 4)?,
                to_u32(output_shape.n, "global mean batch")?,
                1,
            ],
        });
        Ok(output)
    }

    pub fn max_pool(
        &mut self,
        input: Value,
        kernel: [usize; 2],
        stride: [usize; 2],
        output_hw: Option<[usize; 2]>,
    ) -> Result<Value> {
        self.pool(input, kernel, stride, output_hw, true)
    }

    pub fn avg_pool(
        &mut self,
        input: Value,
        kernel: [usize; 2],
        stride: [usize; 2],
    ) -> Result<Value> {
        self.pool(input, kernel, stride, None, false)
    }

    fn pool(
        &mut self,
        input: Value,
        kernel: [usize; 2],
        stride: [usize; 2],
        output_hw: Option<[usize; 2]>,
        max: bool,
    ) -> Result<Value> {
        let output_hw = match output_hw {
            Some(shape) => shape,
            None => [
                conv_output_dim(input.shape.h, kernel[0], stride[0], 0)?,
                conv_output_dim(input.shape.w, kernel[1], stride[1], 0)?,
            ],
        };
        let output_shape = Shape4::new(input.shape.n, output_hw[0], output_hw[1], input.shape.c)?;
        let output = allocate_value(&self.allocator, output_shape)?;
        let params = pool_params(
            &input,
            &output,
            kernel,
            stride,
            [0, 0],
            Activation::None,
            false,
        )?;
        let rows = output_shape
            .n
            .checked_mul(output_shape.h)
            .and_then(|value| value.checked_mul(output_shape.w))
            .ok_or_else(|| Error::InvalidModel("pool output rows overflow".into()))?;
        self.dispatches.push(Dispatch {
            kernel: if max {
                Kernel::PoolMax
            } else {
                Kernel::PoolAvg
            },
            params,
            workgroups: [
                div_ceil_u32(rows, 8)?,
                div_ceil_u32(output_shape.cs / 4, 8)?,
                1,
            ],
        });
        Ok(output)
    }

    pub fn fused_detector_head(
        &mut self,
        input: Value,
        conv: &ConvDesc,
        up: &ConvDesc,
        final_conv: &ConvDesc,
    ) -> Result<Value> {
        let hidden_channels = conv.output_channels;
        if conv.input_channels != input.shape.c
            || conv.kernel != [3, 3]
            || conv.stride != [1, 1]
            || conv.padding != [1, 1]
            || up.input_channels != hidden_channels
            || up.output_channels != hidden_channels
            || final_conv.input_channels != hidden_channels
            || final_conv.output_channels != 1
            || up.kernel != [2, 2]
            || up.stride != [2, 2]
            || final_conv.kernel != [2, 2]
            || final_conv.stride != [2, 2]
            || hidden_channels > 64
        {
            return Err(Error::InvalidModel(
                "unsupported fused detector head configuration".into(),
            ));
        }
        let output_h = input
            .shape
            .h
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidModel("detector head height overflow".into()))?;
        let output_w = input
            .shape
            .w
            .checked_mul(4)
            .ok_or_else(|| Error::InvalidModel("detector head width overflow".into()))?;
        let output_shape = Shape4::new(input.shape.n, output_h, output_w, 1)?;
        let output = allocate_value(&self.allocator, output_shape)?;
        let mut params = [0u32; IMMEDIATE_WORDS];
        params[0] = input.offset();
        params[1] = output.offset();
        params[2] = conv.weight_offset;
        params[3] = conv.bias_offset;
        params[4] = up.weight_offset;
        params[5] = up.bias_offset;
        params[6] = final_conv.weight_offset;
        params[7] = final_conv.bias_offset;
        params[8] = to_u32(input.shape.n, "detector head batch")?;
        params[9] = to_u32(input.shape.h, "detector head input height")?;
        params[10] = to_u32(input.shape.w, "detector head input width")?;
        params[11] = to_u32(input.shape.c, "detector head input channels")?;
        params[12] = to_u32(input.shape.cs, "detector head input stride")?;
        params[13] = to_u32(hidden_channels, "detector head hidden channels")?;
        params[14] = to_u32(output_h, "detector head output height")?;
        params[15] = to_u32(output_w, "detector head output width")?;
        params[16] = u32::from(conv.has_bias)
            | (u32::from(up.has_bias) << 1)
            | (u32::from(final_conv.has_bias) << 2);
        let samples_per_group = 256 / hidden_channels;
        params[18] = to_u32(samples_per_group, "detector head samples per group")?;
        let rows = input
            .shape
            .n
            .checked_mul(input.shape.h)
            .and_then(|value| value.checked_mul(input.shape.w))
            .ok_or_else(|| Error::InvalidModel("detector head rows overflow".into()))?;
        let groups = rows.div_ceil(samples_per_group);
        self.dispatches.push(Dispatch {
            kernel: Kernel::FusedDetectorHead,
            params,
            workgroups: [to_u32(groups, "detector head workgroups")?, 1, 1],
        });
        Ok(output)
    }

    pub fn deconv(
        &mut self,
        input: Value,
        desc: &ConvDesc,
        activation: Activation,
    ) -> Result<Value> {
        if input.shape.c != desc.input_channels
            || desc.depthwise
            || desc.kernel != [2, 2]
            || desc.stride != [2, 2]
            || desc.padding != [0, 0]
        {
            return Err(Error::InvalidModel(
                "GPU deconvolution requires an ungrouped, unpadded 2x2 stride-2 layer".into(),
            ));
        }
        let output_h = input
            .shape
            .h
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidModel("deconv output height overflow".into()))?;
        let output_w = input
            .shape
            .w
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidModel("deconv output width overflow".into()))?;
        let output_shape = Shape4::new(input.shape.n, output_h, output_w, desc.output_channels)?;
        let output = allocate_value(&self.allocator, output_shape)?;
        let mut params = [0u32; IMMEDIATE_WORDS];
        params[0] = input.offset();
        params[1] = output.offset();
        params[2] = INVALID_OFFSET;
        params[3] = desc.weight_offset;
        params[4] = desc.bias_offset;
        params[5] = to_u32(input.shape.n, "batch")?;
        params[6] = to_u32(input.shape.h, "input height")?;
        params[7] = to_u32(input.shape.w, "input width")?;
        params[8] = to_u32(input.shape.c, "input channels")?;
        params[9] = to_u32(input.shape.cs, "input channel stride")?;
        params[10] = to_u32(output_shape.h, "output height")?;
        params[11] = to_u32(output_shape.w, "output width")?;
        params[12] = to_u32(output_shape.c, "output channels")?;
        params[13] = to_u32(output_shape.cs, "output channel stride")?;
        params[14] = 2;
        params[15] = 2;
        params[16] = 2;
        params[17] = 2;
        params[18] = 1;
        params[19] = 1;
        params[20] = to_u32(desc.padding[0], "padding y")?;
        params[21] = to_u32(desc.padding[1], "padding x")?;
        params[22] = to_u32(output_shape.cs, "weight K stride")?;
        params[23] = activation as u32;
        params[24] = u32::from(desc.has_bias);
        let rows = output_shape
            .n
            .checked_mul(output_shape.h)
            .and_then(|value| value.checked_mul(output_shape.w))
            .ok_or_else(|| Error::InvalidModel("deconv output rows overflow".into()))?;
        let (kernel, row_workgroups) = if input.shape.n == 1
            && input.shape.h == 104
            && input.shape.w == 184
            && input.shape.c == 64
            && output_shape.h == 208
            && output_shape.w == 368
            && output_shape.c == 64
            && activation == Activation::Relu
        {
            let input_rows = input
                .shape
                .n
                .checked_mul(input.shape.h)
                .and_then(|value| value.checked_mul(input.shape.w))
                .ok_or_else(|| Error::InvalidModel("deconv input rows overflow".into()))?;
            let groups = div_ceil_u32(input_rows, 32)?
                .checked_mul(4)
                .ok_or_else(|| Error::InvalidModel("deconv phase groups overflow".into()))?;
            (Kernel::DeconvPhase, groups)
        } else if input.shape.n == 1
            && input.shape.h == 208
            && input.shape.w == 368
            && input.shape.c == 64
            && output_shape.h == 416
            && output_shape.w == 736
            && output_shape.c == 1
            && activation == Activation::Sigmoid
        {
            (Kernel::DeconvFinal, div_ceil_u32(rows, 256)?)
        } else {
            (Kernel::Deconv, div_ceil_u32(rows, 8)?)
        };
        self.dispatches.push(Dispatch {
            kernel,
            params,
            workgroups: [row_workgroups, div_ceil_u32(output_shape.c, 32)?, 1],
        });
        Ok(output)
    }

    pub fn softmax(&mut self, input: Value) -> Result<Value> {
        let output = allocate_value(&self.allocator, input.shape)?;
        let rows = input
            .shape
            .n
            .checked_mul(input.shape.h)
            .and_then(|value| value.checked_mul(input.shape.w))
            .ok_or_else(|| Error::InvalidModel("softmax rows overflow".into()))?;
        let mut params = [0u32; IMMEDIATE_WORDS];
        params[0] = input.offset();
        params[1] = output.offset();
        params[2] = INVALID_OFFSET;
        params[3] = to_u32(rows, "softmax rows")?;
        params[4] = to_u32(input.shape.c, "softmax size")?;
        params[5] = to_u32(input.shape.cs, "softmax input stride")?;
        params[6] = to_u32(output.shape.cs, "softmax output stride")?;
        params[7] = 1.0f32.to_bits();
        self.dispatches.push(Dispatch {
            kernel: Kernel::Softmax,
            params,
            workgroups: [to_u32(rows, "softmax workgroups")?, 1, 1],
        });
        Ok(output)
    }

    pub fn layer_norm(
        &mut self,
        input: Value,
        weight_offset: u32,
        bias_offset: u32,
        epsilon: f32,
    ) -> Result<Value> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::InvalidModel(format!(
                "layer normalization epsilon must be finite and positive, got {epsilon}"
            )));
        }
        let output = allocate_value(&self.allocator, input.shape)?;
        let rows = input
            .shape
            .n
            .checked_mul(input.shape.h)
            .and_then(|value| value.checked_mul(input.shape.w))
            .ok_or_else(|| Error::InvalidModel("layer normalization rows overflow".into()))?;
        let mut params = [0u32; IMMEDIATE_WORDS];
        params[0] = input.offset();
        params[1] = output.offset();
        params[2] = weight_offset;
        params[3] = bias_offset;
        params[4] = to_u32(rows, "layer normalization rows")?;
        params[5] = to_u32(input.shape.c, "layer normalization channels")?;
        params[6] = to_u32(input.shape.cs, "layer normalization input stride")?;
        params[7] = to_u32(output.shape.cs, "layer normalization output stride")?;
        params[8] = epsilon.to_bits();
        self.dispatches.push(Dispatch {
            kernel: Kernel::LayerNorm,
            params,
            workgroups: [to_u32(rows, "layer normalization workgroups")?, 1, 1],
        });
        Ok(output)
    }

    pub fn attention(
        &mut self,
        qkv: Value,
        hidden_channels: usize,
        num_heads: usize,
    ) -> Result<Value> {
        if qkv.shape.h != 1 {
            return Err(Error::InvalidModel(format!(
                "attention expects NHWC [N, 1, T, 3C], found height {}",
                qkv.shape.h
            )));
        }
        if hidden_channels == 0 || num_heads == 0 || !hidden_channels.is_multiple_of(num_heads) {
            return Err(Error::InvalidModel(format!(
                "attention hidden channels {hidden_channels} must be nonzero and divisible by nonzero head count {num_heads}"
            )));
        }
        let qkv_channels = hidden_channels
            .checked_mul(3)
            .ok_or_else(|| Error::InvalidModel("attention QKV channels overflow".into()))?;
        if qkv.shape.c != qkv_channels {
            return Err(Error::InvalidModel(format!(
                "attention expects {qkv_channels} QKV channels for hidden size {hidden_channels}, found {}",
                qkv.shape.c
            )));
        }

        let sequence = qkv.shape.w;
        let head_dim = hidden_channels / num_heads;
        let score_shape = Shape4::new(qkv.shape.n, num_heads, sequence, sequence)?;
        let scores = allocate_value(&self.allocator, score_shape)?;
        let output_shape = Shape4::new(qkv.shape.n, 1, sequence, hidden_channels)?;
        let output = allocate_value(&self.allocator, output_shape)?;
        let score_rows = qkv
            .shape
            .n
            .checked_mul(num_heads)
            .and_then(|value| value.checked_mul(sequence))
            .ok_or_else(|| Error::InvalidModel("attention score rows overflow".into()))?;

        let mut params = [0u32; IMMEDIATE_WORDS];
        params[0] = qkv.offset();
        params[1] = scores.offset();
        params[2] = output.offset();
        params[3] = to_u32(qkv.shape.n, "attention batch")?;
        params[4] = to_u32(sequence, "attention sequence")?;
        params[5] = to_u32(hidden_channels, "attention hidden channels")?;
        params[6] = to_u32(qkv.shape.cs, "attention QKV stride")?;
        params[7] = to_u32(score_shape.cs, "attention score stride")?;
        params[8] = to_u32(output_shape.cs, "attention output stride")?;
        params[9] = to_u32(num_heads, "attention heads")?;
        params[10] = to_u32(head_dim, "attention head dimension")?;
        params[11] = (1.0 / (head_dim as f32).sqrt()).to_bits();
        params[12] = to_u32(score_rows, "attention score rows")?;

        let workgroups = [to_u32(score_rows, "attention workgroups")?, 1, 1];
        self.dispatches.push(Dispatch {
            kernel: Kernel::AttentionScores,
            params,
            workgroups,
        });
        self.dispatches.push(Dispatch {
            kernel: Kernel::AttentionContext,
            params,
            workgroups,
        });
        Ok(output)
    }

    pub fn finish(self, output: Value) -> Result<Plan> {
        let arena_elements = self.allocator.borrow().high_water.max(4);
        Ok(Plan {
            input_offset: self.input_offset,
            input_shape: self.input_shape,
            output_offset: output.offset(),
            output_shape: output.shape,
            arena_elements,
            dispatches: self.dispatches,
        })
    }
}

fn elementwise_params(
    left: &Value,
    right: Option<&Value>,
    output: &Value,
    activation: Activation,
    output_shape: Shape4,
) -> Result<[u32; IMMEDIATE_WORDS]> {
    let right_shape = right.map_or(Shape4::new(1, 1, 1, 1)?, |value| value.shape);
    let mut params = [0u32; IMMEDIATE_WORDS];
    params[0] = left.offset();
    params[1] = right.map_or(INVALID_OFFSET, Value::offset);
    params[2] = output.offset();
    params[3] = to_u32(output_shape.n, "elementwise batch")?;
    params[4] = to_u32(left.shape.h, "left height")?;
    params[5] = to_u32(left.shape.w, "left width")?;
    params[6] = to_u32(left.shape.c, "left channels")?;
    params[7] = to_u32(left.shape.cs, "left channel stride")?;
    params[8] = to_u32(right_shape.n, "right batch")?;
    params[9] = to_u32(right_shape.h, "right height")?;
    params[10] = to_u32(right_shape.w, "right width")?;
    params[11] = to_u32(right_shape.c, "right channels")?;
    params[12] = to_u32(right_shape.cs, "right channel stride")?;
    params[13] = to_u32(output_shape.h, "output height")?;
    params[14] = to_u32(output_shape.w, "output width")?;
    params[15] = to_u32(output_shape.c, "output channels")?;
    params[16] = to_u32(output_shape.cs, "output channel stride")?;
    params[17] = activation as u32;
    Ok(params)
}

fn elementwise_workgroups(shape: Shape4) -> Result<u32> {
    let blocks = shape
        .n
        .checked_mul(shape.h)
        .and_then(|value| value.checked_mul(shape.w))
        .and_then(|value| value.checked_mul(shape.cs / 4))
        .ok_or_else(|| Error::InvalidModel("elementwise size overflow".into()))?;
    div_ceil_u32(blocks, 256)
}

fn pool_params(
    input: &Value,
    output: &Value,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    activation: Activation,
    include_padding: bool,
) -> Result<[u32; IMMEDIATE_WORDS]> {
    let mut params = [0u32; IMMEDIATE_WORDS];
    params[0] = input.offset();
    params[1] = output.offset();
    params[2] = to_u32(input.shape.n, "pool batch")?;
    params[3] = to_u32(input.shape.h, "pool input height")?;
    params[4] = to_u32(input.shape.w, "pool input width")?;
    params[5] = to_u32(input.shape.c, "pool input channels")?;
    params[6] = to_u32(input.shape.cs, "pool input stride")?;
    params[7] = to_u32(output.shape.h, "pool output height")?;
    params[8] = to_u32(output.shape.w, "pool output width")?;
    params[9] = to_u32(output.shape.c, "pool output channels")?;
    params[10] = to_u32(output.shape.cs, "pool output stride")?;
    params[11] = to_u32(kernel[0], "pool kernel height")?;
    params[12] = to_u32(kernel[1], "pool kernel width")?;
    params[13] = to_u32(stride[0], "pool stride y")?;
    params[14] = to_u32(stride[1], "pool stride x")?;
    params[15] = 1;
    params[16] = 1;
    params[17] = to_u32(padding[0], "pool padding y")?;
    params[18] = to_u32(padding[1], "pool padding x")?;
    params[19] = activation as u32;
    params[20] = u32::from(include_padding);
    Ok(params)
}

fn allocate_value(allocator: &Rc<RefCell<ArenaAllocator>>, shape: Shape4) -> Result<Value> {
    let (offset, length) = allocator.borrow_mut().allocate(shape.elements()?)?;
    Ok(Value {
        allocation: Rc::new(Allocation {
            offset,
            length,
            allocator: allocator.clone(),
        }),
        shape,
    })
}

fn conv_output_dim(input: usize, kernel: usize, stride: usize, padding: usize) -> Result<usize> {
    if kernel == 0 || stride == 0 {
        return Err(Error::InvalidModel(
            "convolution kernel and stride must be nonzero".into(),
        ));
    }
    input
        .checked_add(
            padding
                .checked_mul(2)
                .ok_or_else(|| Error::InvalidModel("convolution padding overflow".into()))?,
        )
        .and_then(|value| value.checked_sub(kernel))
        .map(|value| value / stride + 1)
        .filter(|value| *value > 0)
        .ok_or_else(|| Error::InvalidModel("convolution produces an empty output".into()))
}

fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidModel(format!("{name} exceeds u32")))
}

fn div_ceil_u32(value: usize, divisor: usize) -> Result<u32> {
    let groups = value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or_else(|| Error::InvalidModel("dispatch size overflow".into()))?;
    to_u32(groups, "dispatch size")
}

pub(crate) struct RawOutput {
    pub shape: Shape4,
    pub values: Vec<f32>,
}

pub(crate) struct Session {
    gpu: Gpu,
    plan: Plan,
    arena: wgpu::Buffer,
    _weights: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    execution_lock: Mutex<()>,
}

impl Session {
    fn new(gpu: Gpu, weights: Vec<f32>, mut plan: Plan) -> Result<Self> {
        let arena_bytes = u64::from(plan.arena_elements)
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| Error::Gpu("activation arena byte size overflow".into()))?;
        let weight_elements = weights.len().max(1);
        let weight_bytes = (weight_elements as u64)
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| Error::Gpu("weight buffer byte size overflow".into()))?;
        let output_bytes = (plan.output_shape.elements()? as u64)
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| Error::Gpu("output byte size overflow".into()))?;
        let device = &gpu.inner.device;
        let max_workgroups = device.limits().max_compute_workgroups_per_dimension;
        split_dispatches(&mut plan.dispatches, max_workgroups)?;
        if let Some(dispatch) = plan.dispatches.iter().find(|dispatch| {
            dispatch
                .workgroups
                .iter()
                .any(|dimension| *dimension > max_workgroups)
        }) {
            return Err(Error::Gpu(format!(
                "{:?} dispatch {:?} exceeds device workgroup-per-dimension limit {max_workgroups}",
                dispatch.kernel, dispatch.workgroups
            )));
        }
        let max_storage_binding = device.limits().max_storage_buffer_binding_size;
        if arena_bytes > device.limits().max_buffer_size
            || weight_bytes > device.limits().max_buffer_size
            || arena_bytes > max_storage_binding
            || weight_bytes > max_storage_binding
        {
            return Err(Error::Gpu(format!(
                "model buffers exceed device limits (max buffer {}, max storage binding {max_storage_binding}, arena {arena_bytes}, weights {weight_bytes})",
                device.limits().max_buffer_size,
            )));
        }
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ppocr activation arena"),
            size: arena_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ppocr weights"),
            size: weight_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !weights.is_empty() {
            gpu.inner
                .queue
                .write_buffer(&weight_buffer, 0, &f32_bytes(&weights));
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ppocr output readback"),
            size: output_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ppocr buffers"),
            layout: &gpu.inner.kernels.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffer.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            gpu,
            plan,
            arena,
            _weights: weight_buffer,
            readback,
            bind_group,
            execution_lock: Mutex::new(()),
        })
    }

    pub fn run_nchw(&self, input: &[f32]) -> Result<RawOutput> {
        let _execution = self
            .execution_lock
            .lock()
            .map_err(|_| Error::Gpu("inference lock is poisoned".into()))?;
        self.upload_nchw(input)?;
        let submission = self.submit(true)?;
        self.gpu
            .inner
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| Error::Gpu(format!("wait for inference: {error}")))?;
        self.read_output()
    }

    pub fn benchmark_nchw(
        &self,
        input: &[f32],
        warmup: usize,
        runs: usize,
    ) -> Result<Vec<std::time::Duration>> {
        let _execution = self
            .execution_lock
            .lock()
            .map_err(|_| Error::Gpu("inference lock is poisoned".into()))?;
        if runs == 0 {
            return Err(Error::InvalidInput("benchmark runs must be nonzero".into()));
        }
        self.upload_nchw(input)?;
        if std::env::var_os("PPOCR_GPU_PROFILE").is_some() {
            self.profile_dispatches()?;
        }
        for _ in 0..warmup {
            let submission = self.submit(false)?;
            self.wait(submission)?;
        }
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = std::time::Instant::now();
            let submission = self.submit(false)?;
            self.wait(submission)?;
            samples.push(start.elapsed());
        }
        Ok(samples)
    }

    fn profile_dispatches(&self) -> Result<()> {
        if !self.gpu.inner.timestamp_profiling {
            return Err(Error::Gpu(
                "the selected adapter does not support timestamps inside compute passes".into(),
            ));
        }
        let query_count = u32::try_from(self.plan.dispatches.len())
            .ok()
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| Error::Gpu("too many dispatches to profile".into()))?;
        let query_bytes = u64::from(query_count) * u64::from(wgpu::QUERY_SIZE);
        let device = &self.gpu.inner.device;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ppocr dispatch timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ppocr timestamp resolve"),
            size: query_bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ppocr timestamp readback"),
            size: query_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ppocr profiled inference"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ppocr profiled graph"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            for (index, dispatch) in self.plan.dispatches.iter().enumerate() {
                let start_query = u32::try_from(index * 2)
                    .map_err(|_| Error::Gpu("profile query index overflow".into()))?;
                pass.write_timestamp(&query_set, start_query);
                pass.set_pipeline(self.gpu.inner.kernels.pipeline(dispatch.kernel));
                pass.set_immediates(0, &words_bytes(&dispatch.params));
                pass.dispatch_workgroups(
                    dispatch.workgroups[0],
                    dispatch.workgroups[1],
                    dispatch.workgroups[2],
                );
                pass.write_timestamp(&query_set, start_query + 1);
            }
        }
        encoder.resolve_query_set(&query_set, 0..query_count, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, Some(query_bytes));
        let submission = self.gpu.inner.queue.submit([encoder.finish()]);
        self.wait(submission)?;

        let (sender, receiver) = mpsc::sync_channel(1);
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Gpu(format!("poll timestamp mapping: {error}")))?;
        receiver
            .recv()
            .map_err(|_| Error::Gpu("timestamp mapping callback was dropped".into()))?
            .map_err(|error| Error::Gpu(format!("map timestamps: {error}")))?;
        let view = readback.slice(..).get_mapped_range();
        let timestamps = view
            .chunks_exact(size_of::<u64>())
            .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("u64 timestamp bytes")))
            .collect::<Vec<_>>();
        let period_ns = f64::from(self.gpu.inner.queue.get_timestamp_period());
        let mut total_ms = 0.0;
        eprintln!("gpu_profile dispatches={}", self.plan.dispatches.len());
        for (index, dispatch) in self.plan.dispatches.iter().enumerate() {
            let elapsed_ms = timestamps[index * 2 + 1].wrapping_sub(timestamps[index * 2]) as f64
                * period_ns
                / 1_000_000.0;
            total_ms += elapsed_ms;
            eprintln!(
                "gpu_profile index={index} kernel={:?} workgroups={:?} input={}x{}x{} output={}x{}x{} kernel_shape={}x{} stride={}x{} ms={elapsed_ms:.6}",
                dispatch.kernel,
                dispatch.workgroups,
                dispatch.params[6],
                dispatch.params[7],
                dispatch.params[8],
                dispatch.params[10],
                dispatch.params[11],
                dispatch.params[12],
                dispatch.params[14],
                dispatch.params[15],
                dispatch.params[16],
                dispatch.params[17],
            );
        }
        eprintln!("gpu_profile total_dispatch_ms={total_ms:.6}");
        drop(view);
        readback.unmap();
        Ok(())
    }

    fn wait(&self, submission: wgpu::SubmissionIndex) -> Result<()> {
        self.gpu
            .inner
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| Error::Gpu(format!("wait for inference: {error}")))?;
        Ok(())
    }

    fn upload_nchw(&self, input: &[f32]) -> Result<()> {
        let expected = self.plan.input_shape.logical_elements()?;
        if input.len() != expected {
            return Err(Error::InvalidInput(format!(
                "input has {} values; expected {expected} for NCHW [{}, {}, {}, {}]",
                input.len(),
                self.plan.input_shape.n,
                self.plan.input_shape.c,
                self.plan.input_shape.h,
                self.plan.input_shape.w
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidInput(
                "input contains non-finite values".into(),
            ));
        }
        let shape = self.plan.input_shape;
        let mut packed = vec![0.0f32; shape.elements()?];
        for n in 0..shape.n {
            for c in 0..shape.c {
                for y in 0..shape.h {
                    for x in 0..shape.w {
                        let source = ((n * shape.c + c) * shape.h + y) * shape.w + x;
                        let target = ((n * shape.h + y) * shape.w + x) * shape.cs + c;
                        packed[target] = input[source];
                    }
                }
            }
        }
        let byte_offset = u64::from(self.plan.input_offset) * size_of::<f32>() as u64;
        self.gpu
            .inner
            .queue
            .write_buffer(&self.arena, byte_offset, &f32_bytes(&packed));
        Ok(())
    }

    fn submit(&self, copy_output: bool) -> Result<wgpu::SubmissionIndex> {
        let device = &self.gpu.inner.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ppocr inference"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ppocr graph"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            for dispatch in &self.plan.dispatches {
                pass.set_pipeline(self.gpu.inner.kernels.pipeline(dispatch.kernel));
                let immediate = words_bytes(&dispatch.params);
                pass.set_immediates(0, &immediate);
                pass.dispatch_workgroups(
                    dispatch.workgroups[0],
                    dispatch.workgroups[1],
                    dispatch.workgroups[2],
                );
            }
        }
        if copy_output {
            let output_bytes = self.plan.output_shape.elements()? as u64 * size_of::<f32>() as u64;
            encoder.copy_buffer_to_buffer(
                &self.arena,
                u64::from(self.plan.output_offset) * size_of::<f32>() as u64,
                &self.readback,
                0,
                output_bytes,
            );
        }
        Ok(self.gpu.inner.queue.submit([encoder.finish()]))
    }

    fn read_output(&self) -> Result<RawOutput> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
        self.gpu
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| Error::Gpu(format!("poll output mapping: {error}")))?;
        receiver
            .recv()
            .map_err(|_| Error::Gpu("output mapping callback was dropped".into()))?
            .map_err(|error| Error::Gpu(format!("map output: {error}")))?;
        let view = self.readback.slice(..).get_mapped_range();
        let packed = bytes_f32(&view)?;
        drop(view);
        self.readback.unmap();
        let shape = self.plan.output_shape;
        let mut values = Vec::with_capacity(shape.logical_elements()?);
        for n in 0..shape.n {
            for y in 0..shape.h {
                for x in 0..shape.w {
                    let base = ((n * shape.h + y) * shape.w + x) * shape.cs;
                    values.extend_from_slice(&packed[base..base + shape.c]);
                }
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Gpu("model output contains non-finite values".into()));
        }
        Ok(RawOutput { shape, values })
    }
}

fn split_dispatches(dispatches: &mut [Dispatch], limit: u32) -> Result<()> {
    if limit == 0 {
        return Err(Error::Gpu(
            "device reports a zero workgroup-per-dimension limit".into(),
        ));
    }

    for dispatch in dispatches {
        if dispatch.workgroups[2] != 1 {
            return Err(Error::Gpu(format!(
                "{:?} logical dispatch must have z=1, found {:?}",
                dispatch.kernel, dispatch.workgroups
            )));
        }
        if matches!(dispatch.kernel, Kernel::GlobalMean) {
            let channel_groups = dispatch.workgroups[0];
            let logical_groups = channel_groups
                .checked_mul(dispatch.workgroups[1])
                .ok_or_else(|| Error::Gpu("global mean dispatch size overflow".into()))?;
            dispatch.params[22] = channel_groups;
            dispatch.workgroups = split_x(logical_groups, 1, limit)?;
            dispatch.params[21] = dispatch.workgroups[0];
            continue;
        }

        dispatch.workgroups = split_x(dispatch.workgroups[0], dispatch.workgroups[1], limit)?;
        dispatch.params[dispatch_x_param(dispatch.kernel)] = dispatch.workgroups[0];
    }
    Ok(())
}

fn split_x(logical_x: u32, y: u32, limit: u32) -> Result<[u32; 3]> {
    if logical_x == 0 || y == 0 {
        return Err(Error::Gpu("dispatch dimensions must be nonzero".into()));
    }
    if y > limit {
        return Err(Error::Gpu(format!(
            "dispatch y dimension {y} exceeds device limit {limit}"
        )));
    }
    let x = logical_x.min(limit);
    let z = logical_x / x + u32::from(!logical_x.is_multiple_of(x));
    if z > limit {
        return Err(Error::Gpu(format!(
            "logical dispatch dimension {logical_x} cannot fit device {limit}x{limit} x/z grid"
        )));
    }
    let physical_groups = u64::from(x) * u64::from(z);
    if physical_groups > u64::from(u32::MAX) + 1 {
        return Err(Error::Gpu(format!(
            "physical dispatch grid {x}x{z} exceeds u32 shader indexing"
        )));
    }
    Ok([x, y, z])
}

const fn dispatch_x_param(kernel: Kernel) -> usize {
    match kernel {
        Kernel::Conv
        | Kernel::Conv2x2Direct
        | Kernel::Conv3x3Direct
        | Kernel::Conv3x3Stride2Direct
        | Kernel::ConvLarge
        | Kernel::ConvSparse9
        | Kernel::ConvSpatialM32
        | Kernel::ConvMediumLinear
        | Kernel::ConvSingleRow
        | Kernel::Depthwise
        | Kernel::Deconv
        | Kernel::DeconvFinal
        | Kernel::DeconvPhase => 25,
        Kernel::FusedDetectorHead => 17,
        Kernel::Add | Kernel::MulChannel | Kernel::ResizeNearest | Kernel::Concat => 19,
        Kernel::GlobalMean | Kernel::PoolMax | Kernel::PoolAvg => 21,
        Kernel::Softmax => 9,
        Kernel::LayerNorm => 9,
        Kernel::AttentionScores | Kernel::AttentionContext => 14,
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn words_bytes(words: &[u32; IMMEDIATE_WORDS]) -> [u8; IMMEDIATE_WORDS * 4] {
    let mut bytes = [0u8; IMMEDIATE_WORDS * 4];
    for (index, word) in words.iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn bytes_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let mut chunks = bytes.chunks_exact(size_of::<f32>());
    let values = chunks
        .by_ref()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    if !chunks.remainder().is_empty() {
        return Err(Error::Gpu("mapped output is not aligned to F32".into()));
    }
    Ok(values)
}

struct Kernels {
    bind_group_layout: wgpu::BindGroupLayout,
    conv: wgpu::ComputePipeline,
    conv_2x2_direct: wgpu::ComputePipeline,
    conv_3x3_direct: wgpu::ComputePipeline,
    conv_3x3_stride2_direct: wgpu::ComputePipeline,
    conv_large: wgpu::ComputePipeline,
    conv_sparse9: wgpu::ComputePipeline,
    conv_spatial_m32: wgpu::ComputePipeline,
    conv_medium_linear: wgpu::ComputePipeline,
    conv_single_row: wgpu::ComputePipeline,
    depthwise: wgpu::ComputePipeline,
    add: wgpu::ComputePipeline,
    mul_channel: wgpu::ComputePipeline,
    resize_nearest: wgpu::ComputePipeline,
    concat: wgpu::ComputePipeline,
    global_mean: wgpu::ComputePipeline,
    pool_max: wgpu::ComputePipeline,
    pool_avg: wgpu::ComputePipeline,
    deconv: wgpu::ComputePipeline,
    deconv_final: wgpu::ComputePipeline,
    deconv_phase: wgpu::ComputePipeline,
    fused_detector_head: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    layer_norm: wgpu::ComputePipeline,
    attention_scores: wgpu::ComputePipeline,
    attention_context: wgpu::ComputePipeline,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Result<Self> {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ppocr kernel buffers"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ppocr kernel layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: IMMEDIATE_BYTES,
        });
        let create = |label: &'static str, source: &'static str, entry: &'static str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        Ok(Self {
            conv: create("ppocr conv", include_str!("shaders/conv.wgsl"), "conv"),
            conv_2x2_direct: create(
                "ppocr direct 2x2 convolution",
                include_str!("shaders/conv_2x2_direct.wgsl"),
                "conv_2x2_direct",
            ),
            conv_3x3_direct: create(
                "ppocr direct 3x3 convolution",
                include_str!("shaders/conv_3x3_direct.wgsl"),
                "conv_3x3_direct",
            ),
            conv_3x3_stride2_direct: create(
                "ppocr direct stride-2 3x3 convolution",
                include_str!("shaders/conv_3x3_stride2_direct.wgsl"),
                "conv_3x3_stride2_direct",
            ),
            conv_large: create(
                "ppocr large convolution",
                include_str!("shaders/conv_large_m32.wgsl"),
                "conv_large_m32",
            ),
            conv_spatial_m32: create(
                "ppocr M32 spatial convolution",
                include_str!("shaders/conv_spatial_m32.wgsl"),
                "conv_spatial_m32",
            ),
            conv_medium_linear: create(
                "ppocr medium linear convolution",
                include_str!("shaders/conv_medium_linear.wgsl"),
                "conv_medium_linear",
            ),
            conv_single_row: create(
                "ppocr single-row convolution",
                include_str!("shaders/conv_single_row.wgsl"),
                "conv_single_row",
            ),
            depthwise: create(
                "ppocr depthwise",
                include_str!("shaders/depthwise.wgsl"),
                "depthwise",
            ),
            add: create(
                "ppocr add",
                include_str!("shaders/elementwise.wgsl"),
                "add_same",
            ),
            mul_channel: create(
                "ppocr mul channel",
                include_str!("shaders/elementwise.wgsl"),
                "mul_broadcast",
            ),
            resize_nearest: create(
                "ppocr resize nearest",
                include_str!("shaders/elementwise.wgsl"),
                "resize_nearest",
            ),
            concat: create(
                "ppocr concat",
                include_str!("shaders/elementwise.wgsl"),
                "concat2",
            ),
            global_mean: create(
                "ppocr global mean",
                include_str!("shaders/pool.wgsl"),
                "global_mean",
            ),
            pool_max: create(
                "ppocr max pool",
                include_str!("shaders/pool.wgsl"),
                "pool_max",
            ),
            pool_avg: create(
                "ppocr avg pool",
                include_str!("shaders/pool.wgsl"),
                "pool_avg",
            ),
            deconv: create(
                "ppocr deconv",
                include_str!("shaders/deconv.wgsl"),
                "deconv2x2",
            ),
            deconv_final: create(
                "ppocr final single-channel deconv",
                include_str!("shaders/deconv_final.wgsl"),
                "deconv_final",
            ),
            deconv_phase: create(
                "ppocr deconv phase GEMM",
                include_str!("shaders/deconv_phase.wgsl"),
                "deconv_phase",
            ),
            conv_sparse9: create(
                "ppocr sparse 9x9 convolution",
                include_str!("shaders/conv_sparse_9x9.wgsl"),
                "conv_sparse_9x9",
            ),
            fused_detector_head: create(
                "ppocr fused detector head",
                include_str!("shaders/detector_head.wgsl"),
                "detector_head",
            ),
            softmax: create(
                "ppocr softmax",
                include_str!("shaders/softmax.wgsl"),
                "softmax",
            ),
            layer_norm: create(
                "ppocr layer norm",
                include_str!("shaders/layer_norm.wgsl"),
                "layer_norm",
            ),
            attention_scores: create(
                "ppocr attention scores",
                include_str!("shaders/attention.wgsl"),
                "attention_scores",
            ),
            attention_context: create(
                "ppocr attention context",
                include_str!("shaders/attention.wgsl"),
                "attention_context",
            ),
            bind_group_layout,
        })
    }

    fn pipeline(&self, kernel: Kernel) -> &wgpu::ComputePipeline {
        match kernel {
            Kernel::Conv => &self.conv,
            Kernel::Conv2x2Direct => &self.conv_2x2_direct,
            Kernel::Conv3x3Direct => &self.conv_3x3_direct,
            Kernel::Conv3x3Stride2Direct => &self.conv_3x3_stride2_direct,
            Kernel::ConvLarge => &self.conv_large,
            Kernel::ConvSparse9 => &self.conv_sparse9,
            Kernel::ConvSpatialM32 => &self.conv_spatial_m32,
            Kernel::ConvMediumLinear => &self.conv_medium_linear,
            Kernel::ConvSingleRow => &self.conv_single_row,
            Kernel::Depthwise => &self.depthwise,
            Kernel::Add => &self.add,
            Kernel::MulChannel => &self.mul_channel,
            Kernel::ResizeNearest => &self.resize_nearest,
            Kernel::Concat => &self.concat,
            Kernel::GlobalMean => &self.global_mean,
            Kernel::PoolMax => &self.pool_max,
            Kernel::PoolAvg => &self.pool_avg,
            Kernel::Deconv => &self.deconv,
            Kernel::DeconvFinal => &self.deconv_final,
            Kernel::DeconvPhase => &self.deconv_phase,
            Kernel::FusedDetectorHead => &self.fused_detector_head,
            Kernel::Softmax => &self.softmax,
            Kernel::LayerNorm => &self.layer_norm,
            Kernel::AttentionScores => &self.attention_scores,
            Kernel::AttentionContext => &self.attention_context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_reuses_and_coalesces_regions() {
        let mut arena = ArenaAllocator::default();
        let (a, _) = arena.allocate(5).unwrap();
        let (b, b_len) = arena.allocate(8).unwrap();
        assert_eq!((a, b), (0, 8));
        arena.release(b, b_len);
        arena.release(a, 8);
        let (whole, _) = arena.allocate(16).unwrap();
        assert_eq!(whole, 0);
        assert_eq!(arena.high_water, 16);
    }

    #[test]
    fn shape_uses_four_channel_padding() {
        let shape = Shape4::new(1, 2, 3, 3).unwrap();
        assert_eq!(shape.cs, 4);
        assert_eq!(shape.elements().unwrap(), 24);
        assert_eq!(shape.logical_elements().unwrap(), 18);
    }

    #[test]
    fn dispatches_split_large_x_dimension_across_z() {
        let mut dispatch = Dispatch {
            kernel: Kernel::Deconv,
            params: [0; IMMEDIATE_WORDS],
            workgroups: [200, 2, 1],
        };
        split_dispatches(std::slice::from_mut(&mut dispatch), 64).unwrap();
        assert_eq!(dispatch.workgroups, [64, 2, 4]);
        assert_eq!(dispatch.params[25], 64);
    }

    #[test]
    fn global_mean_flattens_channel_and_batch_groups() {
        let mut dispatch = Dispatch {
            kernel: Kernel::GlobalMean,
            params: [0; IMMEDIATE_WORDS],
            workgroups: [4, 100, 1],
        };
        split_dispatches(std::slice::from_mut(&mut dispatch), 64).unwrap();
        assert_eq!(dispatch.workgroups, [64, 1, 7]);
        assert_eq!(dispatch.params[21], 64);
        assert_eq!(dispatch.params[22], 4);
    }

    #[test]
    fn dispatch_split_rejects_an_unrepresentable_grid() {
        let mut dispatch = Dispatch {
            kernel: Kernel::Softmax,
            params: [0; IMMEDIATE_WORDS],
            workgroups: [65, 1, 1],
        };
        let error = split_dispatches(std::slice::from_mut(&mut dispatch), 8).unwrap_err();
        assert!(error.to_string().contains("cannot fit"));
    }

    #[test]
    fn dispatch_split_covers_each_logical_group_once() {
        for limit in 1..=8 {
            for logical_x in 1..=limit * limit {
                let [x, y, z] = split_x(logical_x, 1, limit).unwrap();
                assert_eq!(y, 1);
                let mapped = (0..z)
                    .flat_map(|group_z| (0..x).map(move |group_x| group_z * x + group_x))
                    .collect::<Vec<_>>();
                for (expected, found) in mapped[..logical_x as usize].iter().copied().enumerate() {
                    assert_eq!(found, expected as u32);
                }
                assert!(
                    mapped[logical_x as usize..]
                        .iter()
                        .all(|group| *group >= logical_x)
                );
            }
        }
    }

    #[test]
    fn full_hd_detector_rows_fit_xz_grid() {
        let rows = 1_088 * 1_920;
        let logical_x = div_ceil_u32(rows, 8).unwrap();
        let grid = split_x(logical_x, 1, 65_535).unwrap();
        assert_eq!(logical_x, 261_120);
        assert_eq!(grid, [65_535, 1, 4]);
        assert_eq!(3 * grid[0] + 64_514, logical_x - 1);
        assert_eq!(3 * grid[0] + 64_515, logical_x);
    }

    #[test]
    fn dispatch_split_sets_every_kernel_parameter() {
        let cases = [
            (Kernel::Conv, 25),
            (Kernel::Conv2x2Direct, 25),
            (Kernel::Conv3x3Direct, 25),
            (Kernel::Conv3x3Stride2Direct, 25),
            (Kernel::ConvLarge, 25),
            (Kernel::ConvSparse9, 25),
            (Kernel::ConvSpatialM32, 25),
            (Kernel::ConvMediumLinear, 25),
            (Kernel::ConvSingleRow, 25),
            (Kernel::Depthwise, 25),
            (Kernel::Add, 19),
            (Kernel::MulChannel, 19),
            (Kernel::ResizeNearest, 19),
            (Kernel::Concat, 19),
            (Kernel::PoolMax, 21),
            (Kernel::PoolAvg, 21),
            (Kernel::Deconv, 25),
            (Kernel::DeconvFinal, 25),
            (Kernel::DeconvPhase, 25),
            (Kernel::Softmax, 9),
            (Kernel::LayerNorm, 9),
            (Kernel::AttentionScores, 14),
            (Kernel::AttentionContext, 14),
        ];
        for (kernel, param) in cases {
            let mut dispatch = Dispatch {
                kernel,
                params: [0; IMMEDIATE_WORDS],
                workgroups: [65, 1, 1],
            };
            split_dispatches(std::slice::from_mut(&mut dispatch), 64).unwrap();
            assert_eq!(dispatch.workgroups, [64, 1, 2]);
            assert_eq!(dispatch.params[param], 64);
        }
    }

    #[test]
    fn dispatch_split_rejects_preexisting_z_dimension() {
        let mut dispatch = Dispatch {
            kernel: Kernel::Conv,
            params: [0; IMMEDIATE_WORDS],
            workgroups: [1, 1, 2],
        };
        let error = split_dispatches(std::slice::from_mut(&mut dispatch), 64).unwrap_err();
        assert!(error.to_string().contains("must have z=1"));
    }

    #[test]
    fn layer_norm_builds_one_row_dispatch_with_affine_parameters() {
        let (mut builder, input) = GraphBuilder::new([2, 120, 1, 40]).unwrap();
        let output = builder.layer_norm(input, 17, 137, 1e-6).unwrap();
        assert_eq!(output.shape, Shape4::new(2, 1, 40, 120).unwrap());
        assert_eq!(builder.dispatches.len(), 1);
        let dispatch = &builder.dispatches[0];
        assert!(matches!(dispatch.kernel, Kernel::LayerNorm));
        assert_eq!(dispatch.workgroups, [80, 1, 1]);
        assert_eq!(dispatch.params[2], 17);
        assert_eq!(dispatch.params[3], 137);
        assert_eq!(dispatch.params[4..8], [80, 120, 120, 120]);
        assert_eq!(f32::from_bits(dispatch.params[8]), 1e-6);
    }

    #[test]
    fn layer_norm_rejects_nonpositive_or_nonfinite_epsilon() {
        for epsilon in [0.0, -1e-6, f32::NAN, f32::INFINITY] {
            let (mut builder, input) = GraphBuilder::new([1, 4, 1, 1]).unwrap();
            let error = builder.layer_norm(input, 0, 4, epsilon).err().unwrap();
            assert!(error.to_string().contains("finite and positive"));
        }
    }

    #[test]
    fn attention_builds_score_softmax_and_context_dispatches() {
        for hidden_channels in [120, 192] {
            let (mut builder, qkv) = GraphBuilder::new([2, hidden_channels * 3, 1, 40]).unwrap();
            let output = builder.attention(qkv, hidden_channels, 8).unwrap();
            assert_eq!(
                output.shape,
                Shape4::new(2, 1, 40, hidden_channels).unwrap()
            );
            assert_eq!(builder.dispatches.len(), 2);
            assert!(matches!(
                builder.dispatches[0].kernel,
                Kernel::AttentionScores
            ));
            assert!(matches!(
                builder.dispatches[1].kernel,
                Kernel::AttentionContext
            ));
            for dispatch in &builder.dispatches {
                assert_eq!(dispatch.workgroups, [640, 1, 1]);
                assert_eq!(dispatch.params[4], 40);
                assert_eq!(dispatch.params[5], hidden_channels as u32);
                assert_eq!(dispatch.params[6], (hidden_channels * 3) as u32);
                assert_eq!(dispatch.params[7], 40);
                assert_eq!(dispatch.params[8], hidden_channels as u32);
                assert_eq!(dispatch.params[9], 8);
                assert_eq!(dispatch.params[10], (hidden_channels / 8) as u32);
                assert_eq!(dispatch.params[12], 640);
                let scale = f32::from_bits(dispatch.params[11]);
                assert!((scale - 1.0 / (hidden_channels as f32 / 8.0).sqrt()).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn attention_validates_shape_channels_and_heads() {
        let (mut builder, qkv) = GraphBuilder::new([1, 360, 2, 40]).unwrap();
        let error = builder.attention(qkv, 120, 8).err().unwrap();
        assert!(error.to_string().contains("height"));

        let (mut builder, qkv) = GraphBuilder::new([1, 360, 1, 40]).unwrap();
        let error = builder.attention(qkv.clone(), 120, 7).err().unwrap();
        assert!(error.to_string().contains("divisible"));
        let error = builder.attention(qkv.clone(), 0, 8).err().unwrap();
        assert!(error.to_string().contains("nonzero"));
        let error = builder.attention(qkv, 192, 8).err().unwrap();
        assert!(error.to_string().contains("576 QKV channels"));
    }

    #[test]
    fn gpu_layer_norm_and_attention_match_cpu_references() {
        let Ok(gpu) = Gpu::new() else {
            return;
        };

        let (mut builder, input) = GraphBuilder::new([1, 4, 1, 2]).unwrap();
        let output = builder.layer_norm(input, 0, 4, 1e-6).unwrap();
        let session = gpu
            .create_session(
                vec![1.0, 0.5, -1.0, 2.0, 0.1, -0.2, 0.3, -0.4],
                builder.finish(output).unwrap(),
            )
            .unwrap();
        let input_nchw = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        let actual = session.run_nchw(&input_nchw).unwrap().values;
        let mut expected = Vec::new();
        for row in [[1.0f32, 2.0, 3.0, 4.0], [2.0f32, 4.0, 6.0, 8.0]] {
            let mean = row.iter().sum::<f32>() / 4.0;
            let variance = row
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f32>()
                / 4.0;
            for channel in 0..4 {
                let gamma = [1.0, 0.5, -1.0, 2.0][channel];
                let beta = [0.1, -0.2, 0.3, -0.4][channel];
                expected.push((row[channel] - mean) / (variance + 1e-6).sqrt() * gamma + beta);
            }
        }
        assert_close(&actual, &expected, 2e-5);

        for (sequence, hidden, heads) in [(3, 4, 2), (40, 120, 8), (40, 192, 8)] {
            let qkv_nhwc = (0..sequence * hidden * 3)
                .map(|index| ((index * 17) % 101) as f32 * 0.01 - 0.5)
                .collect::<Vec<_>>();
            let mut qkv_nchw = vec![0.0; qkv_nhwc.len()];
            for channel in 0..hidden * 3 {
                for token in 0..sequence {
                    qkv_nchw[channel * sequence + token] = qkv_nhwc[token * hidden * 3 + channel];
                }
            }
            let (mut builder, qkv) = GraphBuilder::new([1, hidden * 3, 1, sequence]).unwrap();
            let output = builder.attention(qkv, hidden, heads).unwrap();
            let session = gpu
                .create_session(Vec::new(), builder.finish(output).unwrap())
                .unwrap();
            let actual = session.run_nchw(&qkv_nchw).unwrap().values;
            let expected = attention_reference(&qkv_nhwc, sequence, hidden, heads);
            assert_close(&actual, &expected, 5e-5);
        }
    }

    fn attention_reference(qkv: &[f32], sequence: usize, hidden: usize, heads: usize) -> Vec<f32> {
        let head_dim = hidden / heads;
        let stride = hidden * 3;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; sequence * hidden];
        for head in 0..heads {
            for query in 0..sequence {
                let mut scores = vec![0.0; sequence];
                for key in 0..sequence {
                    for dim in 0..head_dim {
                        scores[key] += qkv[query * stride + head * head_dim + dim]
                            * qkv[key * stride + hidden + head * head_dim + dim];
                    }
                    scores[key] *= scale;
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - maximum).exp();
                        *score
                    })
                    .sum::<f32>();
                for dim in 0..head_dim {
                    for key in 0..sequence {
                        output[query * hidden + head * head_dim + dim] += scores[key] / denominator
                            * qkv[key * stride + hidden * 2 + head * head_dim + dim];
                    }
                }
            }
        }
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: expected {expected}, found {actual}"
            );
        }
    }
}
