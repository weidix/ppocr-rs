// GPU runtime: NHWC 2x2 stride-2 transposed convolution. Weights are
// [kernel_y, kernel_x, input_channel, output_channel].

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

var<workgroup> input_tile: array<f32, 128>;

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

fn source_value(output_row: u32, input_channel: u32) -> f32 {
    let output_plane = params.output_height * params.output_width;
    let batch_index = output_row / output_plane;
    let output_spatial = output_row - batch_index * output_plane;
    let output_y = output_spatial / params.output_width;
    let output_x = output_spatial - output_y * params.output_width;
    let padded_y = output_y + params.pad_top;
    let padded_x = output_x + params.pad_left;
    let input_y = padded_y / 2u;
    let input_x = padded_x / 2u;
    if input_y >= params.input_height || input_x >= params.input_width {
        return 0.0;
    }
    let input_index = params.input_offset
        + (((batch_index * params.input_height + input_y) * params.input_width + input_x)
        * params.input_channel_stride)
        + input_channel;
    return arena[input_index];
}

fn kernel_index(output_row: u32) -> u32 {
    let output_plane = params.output_height * params.output_width;
    let output_spatial = output_row % output_plane;
    let output_y = output_spatial / params.output_width;
    let output_x = output_spatial - output_y * params.output_width;
    let kernel_y = (output_y + params.pad_top) % 2u;
    let kernel_x = (output_x + params.pad_left) % 2u;
    return kernel_y * 2u + kernel_x;
}

@compute @workgroup_size(8, 8, 1)
fn deconv2x2(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let local_linear = local_id.y * 8u + local_id.x;
    let row_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let output_row = row_group * 8u + local_id.x;
    let output_channel = workgroup_id.y * 32u + local_id.y * 4u;
    let output_rows = params.batch * params.output_height * params.output_width;
    var accum = vec4<f32>(0.0);
    var input_base = 0u;

    loop {
        if input_base >= params.input_channels { break; }

        var tile_index = local_linear;
        loop {
            if tile_index >= 128u { break; }
            let tile_row = tile_index / 16u;
            let tile_channel = tile_index - tile_row * 16u;
            let global_row = row_group * 8u + tile_row;
            let input_channel = input_base + tile_channel;
            var value = 0.0;
            if global_row < output_rows && input_channel < params.input_channels {
                value = source_value(global_row, input_channel);
            }
            input_tile[tile_index] = value;
            tile_index += 64u;
        }

        workgroupBarrier();

        if output_row < output_rows {
            let kernel = kernel_index(output_row);
            for (var tile_channel = 0u; tile_channel < 16u; tile_channel += 1u) {
                let input_channel = input_base + tile_channel;
                if input_channel < params.input_channels {
                    let weight_base = params.weight_offset
                        + (kernel * params.input_channels + input_channel) * params.weight_k_stride
                        + output_channel;
                    var weight_value = vec4<f32>(0.0);
                    for (var lane = 0u; lane < 4u; lane += 1u) {
                        if output_channel + lane < params.output_channels {
                            weight_value[lane] = weights[weight_base + lane];
                        }
                    }
                    accum += input_tile[local_id.x * 16u + tile_channel] * weight_value;
                }
            }
        }

        workgroupBarrier();
        input_base += 16u;
    }

    if output_row >= output_rows { return; }
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
        } else if channel < params.output_channel_stride {
            arena[output_base + lane] = 0.0;
        }
    }
}
