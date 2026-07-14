// GPU runtime: NHWC depthwise convolution. Weights use [kernel_y, kernel_x, channel]
// with weight_k_stride values per kernel position.

struct DepthwiseParams {
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

var<immediate> params: DepthwiseParams;

fn sigmoid_scalar(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

fn activate_scalar(value: f32, code: u32) -> f32 {
    switch code {
        case 1u: { return max(value, 0.0); }
        case 2u: { return value * sigmoid_scalar(value); }
        case 3u: { return clamp(value / 6.0 + 0.5, 0.0, 1.0); }
        case 4u: { return clamp(value / 5.0 + 0.5, 0.0, 1.0); }
        case 5u: {
            let scaled = value * 0.7071067811865476;
            let magnitude = abs(scaled);
            let t = 1.0 / (1.0 + 0.3275911 * magnitude);
            let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t;
            let erf = select(-1.0, 1.0, scaled >= 0.0) * (1.0 - polynomial * exp(-magnitude * magnitude));
            return 0.5 * value * (1.0 + erf);
        }
        case 6u: { return value * clamp(value / 6.0 + 0.5, 0.0, 1.0); }
        case 7u: { return sigmoid_scalar(value); }
        default: { return value; }
    }
}

@compute @workgroup_size(8, 8, 1)
fn depthwise(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let output_row = row_group * 8u + local_id.x;
    let output_channel = (workgroup_id.y * 8u + local_id.y) * 4u;
    let output_plane = params.output_height * params.output_width;
    let output_rows = params.batch * output_plane;
    if output_row >= output_rows || output_channel >= params.output_channel_stride {
        return;
    }

    let batch_index = output_row / output_plane;
    let output_spatial = output_row - batch_index * output_plane;
    let output_y = output_spatial / params.output_width;
    let output_x = output_spatial - output_y * params.output_width;
    var accum = vec4<f32>(0.0);

    for (var kernel_y = 0u; kernel_y < params.kernel_height; kernel_y += 1u) {
        let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) - i32(params.pad_top);
        if input_y < 0 || input_y >= i32(params.input_height) { continue; }

        for (var kernel_x = 0u; kernel_x < params.kernel_width; kernel_x += 1u) {
            let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) - i32(params.pad_left);
            if input_x < 0 || input_x >= i32(params.input_width) { continue; }

            let input_base = params.input_offset
                + (((batch_index * params.input_height + u32(input_y)) * params.input_width + u32(input_x))
                * params.input_channel_stride)
                + output_channel;
            let weight_base = params.weight_offset
                + (kernel_y * params.kernel_width + kernel_x) * params.weight_k_stride
                + output_channel;

            for (var lane = 0u; lane < 4u; lane += 1u) {
                let channel = output_channel + lane;
                if channel < params.input_channels && channel < params.output_channels {
                    accum[lane] += arena[input_base + lane] * weights[weight_base + lane];
                }
            }
        }
    }

    let output_base = params.output_offset + output_row * params.output_channel_stride + output_channel;
    for (var lane = 0u; lane < 4u; lane += 1u) {
        let channel = output_channel + lane;
        if channel < params.output_channels {
            var value = accum[lane];
            if (params.flags & 1u) != 0u {
                value += weights[params.bias_offset + channel];
            }
            if (params.flags & 2u) != 0u {
                value += arena[params.add_offset + output_row * params.output_channel_stride + channel];
            }
            arena[output_base + lane] = activate_scalar(value, params.activation);
        } else {
            arena[output_base + lane] = 0.0;
        }
    }
}
