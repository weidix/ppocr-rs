// GPU runtime: specialized NHWC 1x1 convolution for the medium recognizer backbone.
// The selected layers have input/output channel counts divisible by 256.

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

var<immediate> params: ConvParams;

// The padded M16 x K16 input plus K16 x N128 weights use 9.0625 KiB.
// A 17-value row stride avoids same-bank reads across adjacent output rows.
var<workgroup> input_tile: array<f32, 272>;
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

@compute @workgroup_size(8, 16, 1)
fn conv_medium_linear(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let output_rows = params.batch * params.output_height * params.output_width;
    let row_base = row_group * 16u;
    if row_base >= output_rows {
        return;
    }

    let local_linear = local_id.y * 8u + local_id.x;
    let output_channel = workgroup_id.y * 128u + local_id.y * 8u;
    let row0 = row_base + local_id.x;
    let row1 = row0 + 8u;
    var accum00 = vec4<f32>(0.0);
    var accum01 = vec4<f32>(0.0);
    var accum10 = vec4<f32>(0.0);
    var accum11 = vec4<f32>(0.0);
    var k_base = 0u;

    loop {
        if k_base >= params.input_channels { break; }

        var tile_index = local_linear;
        loop {
            if tile_index >= 256u { break; }
            let tile_row = tile_index / 16u;
            let tile_k = tile_index - tile_row * 16u;
            let global_row = row_base + tile_row;
            var value = 0.0;
            if global_row < output_rows {
                value = arena[params.input_offset
                    + global_row * params.input_channel_stride
                    + k_base + tile_k];
            }
            input_tile[tile_row * 17u + tile_k] = value;
            tile_index += 128u;
        }

        var weight_index = local_linear;
        loop {
            if weight_index >= 512u { break; }
            let tile_k = weight_index / 32u;
            let channel4 = weight_index - tile_k * 32u;
            let source = params.weight_offset
                + (k_base + tile_k) * params.weight_k_stride
                + workgroup_id.y * 128u
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

        let weight_lane = local_id.y * 2u;
        let input_base0 = local_id.x * 17u;
        let input_base1 = (local_id.x + 8u) * 17u;
        for (var k = 0u; k < 16u; k += 1u) {
            let weight_base = k * 32u + weight_lane;
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
        k_base += 16u;
    }

    if row0 < output_rows {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(row0, output_channel + lane, accum00[lane]);
            write_output(row0, output_channel + lane + 4u, accum01[lane]);
        }
    }
    if row1 < output_rows {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(row1, output_channel + lane, accum10[lane]);
            write_output(row1, output_channel + lane + 4u, accum11[lane]);
        }
    }
}
