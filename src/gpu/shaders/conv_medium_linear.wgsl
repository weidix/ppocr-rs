// GPU runtime: vectorized NHWC 1x1 convolution for F32 channel-aligned layers.

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
var<storage, read_write> arena: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> weights: array<vec4<f32>>;

var<immediate> params: ConvParams;

// The padded M64 x K32 input plus K32 x N64 weights use 16.25 KiB.
// A 33-value row stride avoids same-bank reads across adjacent output rows.
var<workgroup> input_tile: array<f32, 2112>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

fn sigmoid_value(value: vec4<f32>) -> vec4<f32> {
    return 1.0 / (1.0 + exp(-value));
}

fn activate_value(value: vec4<f32>, code: u32) -> vec4<f32> {
    switch code {
        case 1u: { return max(value, vec4<f32>(0.0)); }
        case 2u: { return value * sigmoid_value(value); }
        case 3u: { return clamp(value / 6.0 + 0.5, vec4<f32>(0.0), vec4<f32>(1.0)); }
        case 4u: { return clamp(value / 5.0 + 0.5, vec4<f32>(0.0), vec4<f32>(1.0)); }
        case 5u: {
            let scaled = value * 0.7071067811865476;
            let magnitude = abs(scaled);
            let t = 1.0 / (1.0 + 0.3275911 * magnitude);
            let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t;
            let erf = select(vec4<f32>(-1.0), vec4<f32>(1.0), scaled >= vec4<f32>(0.0))
                * (1.0 - polynomial * exp(-magnitude * magnitude));
            return 0.5 * value * (1.0 + erf);
        }
        case 6u: {
            return value * clamp(value / 6.0 + 0.5, vec4<f32>(0.0), vec4<f32>(1.0));
        }
        case 7u: { return sigmoid_value(value); }
        default: { return value; }
    }
}

fn write_output(output_row: u32, channel: u32, value: vec4<f32>) {
    if channel >= params.output_channel_stride {
        return;
    }
    var result = value;
    if (params.flags & 1u) != 0u {
        result += weights[(params.bias_offset + channel) / 4u];
    }
    if (params.flags & 2u) != 0u {
        result += arena[(params.add_offset + output_row * params.output_channel_stride + channel) / 4u];
    }
    arena[(params.output_offset + output_row * params.output_channel_stride + channel) / 4u] =
        activate_value(result, params.activation);
}

@compute @workgroup_size(16, 16, 1)
fn conv_medium_linear(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let output_rows = params.batch * params.output_height * params.output_width;
    let row_base = row_group * 64u;
    if row_base >= output_rows {
        return;
    }

    let local_linear = local_id.y * 16u + local_id.x;
    let output_channel = workgroup_id.y * 64u + local_id.y * 4u;
    let row0 = row_base + local_id.x;
    let row1 = row0 + 16u;
    let row2 = row0 + 32u;
    let row3 = row0 + 48u;
    var accum00 = vec4<f32>(0.0);
    var accum10 = vec4<f32>(0.0);
    var accum20 = vec4<f32>(0.0);
    var accum30 = vec4<f32>(0.0);
    var k_base = 0u;

    loop {
        if k_base >= params.input_channels { break; }

        var tile_index = local_linear;
        loop {
            if tile_index >= 512u { break; }
            let tile_row = tile_index / 8u;
            let tile_k4 = tile_index - tile_row * 8u;
            let global_row = row_base + tile_row;
            var value = vec4<f32>(0.0);
            if global_row < output_rows && k_base + tile_k4 * 4u < params.input_channels {
                value = arena[(params.input_offset
                    + global_row * params.input_channel_stride
                    + k_base + tile_k4 * 4u) / 4u];
            }
            let destination = tile_row * 33u + tile_k4 * 4u;
            input_tile[destination] = value.x;
            input_tile[destination + 1u] = value.y;
            input_tile[destination + 2u] = value.z;
            input_tile[destination + 3u] = value.w;
            tile_index += 256u;
        }

        var weight_index = local_linear;
        loop {
            if weight_index >= 512u { break; }
            let tile_k = weight_index / 16u;
            let channel4 = weight_index - tile_k * 16u;
            let source = params.weight_offset
                + (k_base + tile_k) * params.weight_k_stride
                + workgroup_id.y * 64u
                + channel4 * 4u;
            let global_channel = workgroup_id.y * 64u + channel4 * 4u;
            var value = vec4<f32>(0.0);
            if k_base + tile_k < params.input_channels
                && global_channel < params.output_channel_stride {
                value = weights[source / 4u];
            }
            weight_tile[weight_index] = value;
            weight_index += 256u;
        }

        workgroupBarrier();

        let weight_lane = local_id.y;
        let input_base0 = local_id.x * 33u;
        let input_base1 = (local_id.x + 16u) * 33u;
        let input_base2 = (local_id.x + 32u) * 33u;
        let input_base3 = (local_id.x + 48u) * 33u;
        var partial00 = vec4<f32>(0.0);
        var partial10 = vec4<f32>(0.0);
        var partial20 = vec4<f32>(0.0);
        var partial30 = vec4<f32>(0.0);
        for (var k = 0u; k < 16u; k += 1u) {
            let weight_base = k * 16u + weight_lane;
            let weight0 = weight_tile[weight_base];
            let input0 = input_tile[input_base0 + k];
            let input1 = input_tile[input_base1 + k];
            let input2 = input_tile[input_base2 + k];
            let input3 = input_tile[input_base3 + k];
            partial00 += input0 * weight0;
            partial10 += input1 * weight0;
            partial20 += input2 * weight0;
            partial30 += input3 * weight0;
        }
        accum00 += partial00;
        accum10 += partial10;
        accum20 += partial20;
        accum30 += partial30;

        partial00 = vec4<f32>(0.0);
        partial10 = vec4<f32>(0.0);
        partial20 = vec4<f32>(0.0);
        partial30 = vec4<f32>(0.0);
        for (var k = 16u; k < 32u; k += 1u) {
            let weight_base = k * 16u + weight_lane;
            let weight0 = weight_tile[weight_base];
            let input0 = input_tile[input_base0 + k];
            let input1 = input_tile[input_base1 + k];
            let input2 = input_tile[input_base2 + k];
            let input3 = input_tile[input_base3 + k];
            partial00 += input0 * weight0;
            partial10 += input1 * weight0;
            partial20 += input2 * weight0;
            partial30 += input3 * weight0;
        }
        accum00 += partial00;
        accum10 += partial10;
        accum20 += partial20;
        accum30 += partial30;

        workgroupBarrier();
        k_base += 32u;
    }

    if row0 < output_rows {
        write_output(row0, output_channel, accum00);
    }
    if row1 < output_rows {
        write_output(row1, output_channel, accum10);
    }
    if row2 < output_rows {
        write_output(row2, output_channel, accum20);
    }
    if row3 < output_rows {
        write_output(row3, output_channel, accum30);
    }
}
