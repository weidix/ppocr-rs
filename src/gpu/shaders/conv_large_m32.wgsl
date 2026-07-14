// GPU runtime: specialized NHWC 9x9, stride-1, pad-4 convolution with 64 output channels.
// A 32-pixel row tile reuses each Kx64 weight tile twice as much as M16.

struct ConvParams {
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
    reserved0: u32,
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

var<immediate> params: ConvParams;

// Padding the K dimension breaks same-bank accesses when adjacent lanes read
// the same K value from different output rows.
var<workgroup> input_tile: array<f32, 1056>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

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
            let erf = select(-1.0, 1.0, scaled >= 0.0)
                * (1.0 - polynomial * exp(-magnitude * magnitude));
            return 0.5 * value * (1.0 + erf);
        }
        case 6u: { return value * clamp(value / 6.0 + 0.5, 0.0, 1.0); }
        case 7u: { return sigmoid_scalar(value); }
        default: { return value; }
    }
}

fn write_output(output_row: u32, channel: u32, value: f32) {
    var result = value;
    if (params.flags & 1u) != 0u {
        result += weights[params.bias_offset + channel];
    }
    if (params.flags & 2u) != 0u {
        result += arena[params.add_offset + output_row * params.output_channel_stride + channel];
    }
    arena[params.output_offset + output_row * params.output_channel_stride + channel] =
        activate_scalar(result, params.activation);
}

@compute @workgroup_size(16, 8, 1)
fn conv_large_m32(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.reserved0 + workgroup_id.x;
    let tiles_per_row = params.reserved1;
    let row_tile = params.batch * params.output_height * tiles_per_row;
    if row_group >= row_tile {
        return;
    }

    let local_linear = local_id.y * 16u + local_id.x;
    let tile_x = row_group % tiles_per_row;
    let output_yb = row_group / tiles_per_row;
    let batch_index = output_yb / params.output_height;
    let output_y = output_yb - batch_index * params.output_height;
    let output_x0 = tile_x * 32u + local_id.x;
    let output_x1 = output_x0 + 16u;
    let output_row0 = (batch_index * params.output_height + output_y) * params.output_width + output_x0;
    let output_row1 = output_row0 + 16u;
    let output_channel = local_id.y * 8u;
    var accum00 = vec4<f32>(0.0);
    var accum01 = vec4<f32>(0.0);
    var accum10 = vec4<f32>(0.0);
    var accum11 = vec4<f32>(0.0);
    for (var kernel_y = 0u; kernel_y < 9u; kernel_y += 1u) {
        for (var kernel_x = 0u; kernel_x < 9u; kernel_x += 1u) {
            for (var channel_base = 0u; channel_base < params.input_channels; channel_base += 32u) {
                let tile_k = local_linear & 31u;
                let input_channel = channel_base + tile_k;
                var tile_row = local_linear / 32u;
                loop {
                    if tile_row >= 32u { break; }
                    let output_x = tile_x * 32u + tile_row;
                    var value = 0.0;
                    if output_x < params.output_width {
                        let input_y = i32(output_y + kernel_y) - 4;
                        let input_x = i32(output_x + kernel_x) - 4;
                        if input_y >= 0 && input_x >= 0
                            && input_y < i32(params.input_height)
                            && input_x < i32(params.input_width) {
                            let input_index = params.input_offset
                                + (((batch_index * params.input_height + u32(input_y)) * params.input_width
                                + u32(input_x)) * params.input_channel_stride)
                                + input_channel;
                            value = arena[input_index];
                        }
                    }
                    input_tile[tile_row * 33u + tile_k] = value;
                    tile_row += 4u;
                }

                let weight_k_base = (kernel_y * 9u + kernel_x) * params.input_channels
                    + channel_base;
                var weight_index = local_linear;
                loop {
                    if weight_index >= 512u { break; }
                    let tile_weight_k = weight_index / 16u;
                    let channel4 = weight_index - tile_weight_k * 16u;
                    let source = params.weight_offset
                        + (weight_k_base + tile_weight_k) * params.weight_k_stride
                        + channel4 * 4u;
                    weight_tile[weight_index] = vec4<f32>(
                        weights[source],
                        weights[source + 1u],
                        weights[source + 2u],
                        weights[source + 3u],
                    );
                    weight_index += 128u;
                }

                workgroupBarrier();

                let input_base0 = local_id.x * 33u;
                let input_base1 = (local_id.x + 16u) * 33u;
                let weight_lane = local_id.y * 2u;
                for (var k = 0u; k < 32u; k += 1u) {
                    let weight_base = k * 16u + weight_lane;
                    let weight0 = weight_tile[weight_base];
                    let weight1 = weight_tile[weight_base + 1u];
                    let input0 = input_tile[input_base0 + k];
                    let input1 = input_tile[input_base1 + k];
                    accum00 += input0 * weight0;
                    accum01 += input0 * weight1;
                    accum10 += input1 * weight0;
                    accum11 += input1 * weight1;
                }

                workgroupBarrier();
            }
        }
    }

    if output_x0 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row0, output_channel + lane, accum00[lane]);
            write_output(output_row0, output_channel + lane + 4u, accum01[lane]);
        }
    }
    if output_x1 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row1, output_channel + lane, accum10[lane]);
            write_output(output_row1, output_channel + lane + 4u, accum11[lane]);
        }
    }
}
