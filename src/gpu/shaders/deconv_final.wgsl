// GPU runtime: specialized final NHWC 2x2 stride-2 transposed convolution with one output channel.
// Each invocation computes one output pixel; no shared memory or barriers are needed.

struct DeconvParams {
    input_offset: u32,
    output_offset: u32,
    add_offset: u32,
    weight_offset: u32,
    bias_offset: u32,
    batch: u32,
    input_height: u32,
    input_width: u32,
    input_channels: u32,
    input_channel_stride: u32,
    output_height: u32,
    output_width: u32,
    output_channels: u32,
    output_channel_stride: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride_y: u32,
    stride_x: u32,
    dilation_y: u32,
    dilation_x: u32,
    pad_top: u32,
    pad_left: u32,
    weight_k_stride: u32,
    activation: u32,
    flags: u32,
    dispatch_x: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
    reserved4: u32,
    reserved5: u32,
    reserved6: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: DeconvParams;

@compute @workgroup_size(256, 1, 1)
fn deconv_final(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let output_row = row_group * 256u + local_index;
    let output_plane = params.output_height * params.output_width;
    let output_rows = params.batch * output_plane;
    if output_row >= output_rows {
        return;
    }

    let batch_index = output_row / output_plane;
    let output_spatial = output_row - batch_index * output_plane;
    let output_y = output_spatial / params.output_width;
    let output_x = output_spatial - output_y * params.output_width;
    let input_y = output_y / 2u;
    let input_x = output_x / 2u;
    let phase = (output_y & 1u) * 2u + (output_x & 1u);
    let input_base = params.input_offset
        + (((batch_index * params.input_height + input_y) * params.input_width + input_x)
        * params.input_channel_stride);
    let weight_base = params.weight_offset + phase * params.input_channels * params.weight_k_stride;
    var accum = 0.0;
    for (var input_channel = 0u; input_channel < params.input_channels; input_channel += 1u) {
        accum += arena[input_base + input_channel]
            * weights[weight_base + input_channel * params.weight_k_stride];
    }
    if (params.flags & 1u) != 0u {
        accum += weights[params.bias_offset];
    }
    arena[params.output_offset + output_row * params.output_channel_stride] =
        1.0 / (1.0 + exp(-accum));
}
