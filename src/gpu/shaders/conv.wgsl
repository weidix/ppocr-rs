// GPU runtime: NHWC ungrouped convolution. Weights are K-major, where
// K = ((kernel_y * kernel_width + kernel_x) * input_channels + input_channel).
// Each K row has weight_k_stride output-channel values.

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

fn load_weight(index: u32) -> f32 {
    return weights[index];
}

var<immediate> params: ConvParams;

// The common path uses an 8x32 output tile. Very wide projections use a
// 16x64 tile to reuse each weight row across twice as many output rows.
var<workgroup> input_tile: array<f32, 512>;
var<workgroup> weight_tile: array<f32, 1024>;

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

fn im2col_value(output_row: u32, k: u32) -> f32 {
    let output_plane = params.output_height * params.output_width;
    let batch_index = output_row / output_plane;
    let output_spatial = output_row - batch_index * output_plane;
    let output_y = output_spatial / params.output_width;
    let output_x = output_spatial - output_y * params.output_width;

    let input_channel = k % params.input_channels;
    let kernel_spatial = k / params.input_channels;
    let kernel_y = kernel_spatial / params.kernel_width;
    let kernel_x = kernel_spatial - kernel_y * params.kernel_width;
    let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) - i32(params.pad_top);
    let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) - i32(params.pad_left);

    if input_y < 0 || input_x < 0 || input_y >= i32(params.input_height) || input_x >= i32(params.input_width) {
        return 0.0;
    }

    let input_index = params.input_offset
        + (((batch_index * params.input_height + u32(input_y)) * params.input_width + u32(input_x))
        * params.input_channel_stride)
        + input_channel;
    return arena[input_index];
}

fn write_conv_output(output_row: u32, channel: u32, value: f32) {
    let output_index = params.output_offset + output_row * params.output_channel_stride + channel;
    if channel < params.output_channels {
        var result = value;
        if (params.flags & 1u) != 0u {
            result += load_weight(params.bias_offset + channel);
        }
        if (params.flags & 2u) != 0u {
            result += arena[params.add_offset + output_row * params.output_channel_stride + channel];
        }
        arena[output_index] = activate_scalar(result, params.activation);
    } else if channel < params.output_channel_stride {
        arena[output_index] = 0.0;
    }
}

fn conv_wide(
    local_id: vec3<u32>,
    workgroup_id: vec3<u32>,
    output_rows: u32,
    k_size: u32,
) {
    let row_group = workgroup_id.z * params.reserved0 + workgroup_id.x;
    let row_tile = (output_rows + 15u) / 16u;
    let channel_tile = (params.output_channels + 63u) / 64u;
    if row_group >= row_tile || workgroup_id.y >= channel_tile {
        return;
    }

    let local_linear = local_id.y * 8u + local_id.x;
    let output_row0 = row_group * 16u + local_id.x;
    let output_row1 = output_row0 + 8u;
    let output_channel = workgroup_id.y * 64u + local_id.y * 8u;
    var accum00 = vec4<f32>(0.0);
    var accum01 = vec4<f32>(0.0);
    var accum10 = vec4<f32>(0.0);
    var accum11 = vec4<f32>(0.0);
    var k_base = 0u;

    loop {
        if k_base >= k_size { break; }

        var tile_index = local_linear;
        loop {
            if tile_index >= 256u { break; }
            let tile_row = tile_index / 16u;
            let tile_k = tile_index - tile_row * 16u;
            let global_row = row_group * 16u + tile_row;
            let global_k = k_base + tile_k;
            var value = 0.0;
            if global_row < output_rows && global_k < k_size {
                value = im2col_value(global_row, global_k);
            }
            input_tile[tile_index] = value;
            tile_index += 64u;
        }

        tile_index = local_linear;
        loop {
            if tile_index >= 1024u { break; }
            let tile_k = tile_index / 64u;
            let tile_channel = tile_index - tile_k * 64u;
            let global_k = k_base + tile_k;
            let global_channel = workgroup_id.y * 64u + tile_channel;
            var value = 0.0;
            if global_k < k_size && global_channel < params.output_channels {
                value = load_weight(params.weight_offset + global_k * params.weight_k_stride + global_channel);
            }
            weight_tile[tile_index] = value;
            tile_index += 64u;
        }

        workgroupBarrier();

        let input_base0 = local_id.x * 16u;
        let input_base1 = (local_id.x + 8u) * 16u;
        let weight_lane = local_id.y * 8u;
        var partial00 = vec4<f32>(0.0);
        var partial01 = vec4<f32>(0.0);
        var partial10 = vec4<f32>(0.0);
        var partial11 = vec4<f32>(0.0);
        for (var tile_k = 0u; tile_k < 16u; tile_k += 1u) {
            let weight_base = tile_k * 64u + weight_lane;
            let weight0 = vec4<f32>(
                weight_tile[weight_base],
                weight_tile[weight_base + 1u],
                weight_tile[weight_base + 2u],
                weight_tile[weight_base + 3u],
            );
            let weight1 = vec4<f32>(
                weight_tile[weight_base + 4u],
                weight_tile[weight_base + 5u],
                weight_tile[weight_base + 6u],
                weight_tile[weight_base + 7u],
            );
            let input0 = input_tile[input_base0 + tile_k];
            let input1 = input_tile[input_base1 + tile_k];
            partial00 += input0 * weight0;
            partial01 += input0 * weight1;
            partial10 += input1 * weight0;
            partial11 += input1 * weight1;
        }
        accum00 += partial00;
        accum01 += partial01;
        accum10 += partial10;
        accum11 += partial11;

        workgroupBarrier();
        k_base += 16u;
    }

    if output_row0 < output_rows {
        for (var lane = 0u; lane < 8u; lane += 1u) {
            var value = 0.0;
            if lane < 4u { value = accum00[lane]; } else { value = accum01[lane - 4u]; }
            write_conv_output(output_row0, output_channel + lane, value);
        }
    }
    if output_row1 < output_rows {
        for (var lane = 0u; lane < 8u; lane += 1u) {
            var value = 0.0;
            if lane < 4u { value = accum10[lane]; } else { value = accum11[lane - 4u]; }
            write_conv_output(output_row1, output_channel + lane, value);
        }
    }
}

fn conv_1x1_m32(
    local_id: vec3<u32>,
    workgroup_id: vec3<u32>,
    output_rows: u32,
) {
    let row_group = workgroup_id.z * params.reserved0 + workgroup_id.x;
    let row_tile = (output_rows + 31u) / 32u;
    if row_group >= row_tile {
        return;
    }

    let local_linear = local_id.y * 8u + local_id.x;
    let output_row0 = row_group * 32u + local_id.x;
    let output_row1 = output_row0 + 8u;
    let output_row2 = output_row0 + 16u;
    let output_row3 = output_row0 + 24u;
    let output_channel = workgroup_id.y * 32u + local_id.y * 4u;
    var accum0 = vec4<f32>(0.0);
    var accum1 = vec4<f32>(0.0);
    var accum2 = vec4<f32>(0.0);
    var accum3 = vec4<f32>(0.0);
    var k_base = 0u;

    loop {
        if k_base >= params.input_channels { break; }

        var tile_index = local_linear;
        loop {
            if tile_index >= 512u { break; }
            let tile_row = tile_index / 16u;
            let tile_k = tile_index - tile_row * 16u;
            let global_row = row_group * 32u + tile_row;
            let global_k = k_base + tile_k;
            var value = 0.0;
            if global_row < output_rows && global_k < params.input_channels {
                value = arena[params.input_offset + global_row * params.input_channel_stride + global_k];
            }
            input_tile[tile_index] = value;
            tile_index += 64u;
        }

        tile_index = local_linear;
        loop {
            if tile_index >= 512u { break; }
            let tile_k = tile_index / 32u;
            let tile_channel = tile_index - tile_k * 32u;
            let global_k = k_base + tile_k;
            let global_channel = workgroup_id.y * 32u + tile_channel;
            var value = 0.0;
            if global_k < params.input_channels && global_channel < params.output_channels {
                value = load_weight(params.weight_offset + global_k * params.weight_k_stride + global_channel);
            }
            weight_tile[tile_index] = value;
            tile_index += 64u;
        }

        workgroupBarrier();

        let weight_lane = local_id.y * 4u;
        let input_base0 = local_id.x * 16u;
        let input_base1 = (local_id.x + 8u) * 16u;
        let input_base2 = (local_id.x + 16u) * 16u;
        let input_base3 = (local_id.x + 24u) * 16u;
        var partial0 = vec4<f32>(0.0);
        var partial1 = vec4<f32>(0.0);
        var partial2 = vec4<f32>(0.0);
        var partial3 = vec4<f32>(0.0);
        for (var tile_k = 0u; tile_k < 16u; tile_k += 1u) {
            let weight_base = tile_k * 32u + weight_lane;
            let weight_value = vec4<f32>(
                weight_tile[weight_base],
                weight_tile[weight_base + 1u],
                weight_tile[weight_base + 2u],
                weight_tile[weight_base + 3u],
            );
            partial0 += input_tile[input_base0 + tile_k] * weight_value;
            partial1 += input_tile[input_base1 + tile_k] * weight_value;
            partial2 += input_tile[input_base2 + tile_k] * weight_value;
            partial3 += input_tile[input_base3 + tile_k] * weight_value;
        }
        accum0 += partial0;
        accum1 += partial1;
        accum2 += partial2;
        accum3 += partial3;

        workgroupBarrier();
        k_base += 16u;
    }

    if output_row0 < output_rows {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_conv_output(output_row0, output_channel + lane, accum0[lane]);
        }
    }
    if output_row1 < output_rows {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_conv_output(output_row1, output_channel + lane, accum1[lane]);
        }
    }
    if output_row2 < output_rows {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_conv_output(output_row2, output_channel + lane, accum2[lane]);
        }
    }
    if output_row3 < output_rows {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_conv_output(output_row3, output_channel + lane, accum3[lane]);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn conv(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.reserved0 + workgroup_id.x;
    let local_linear = local_id.y * 8u + local_id.x;
    let output_row = row_group * 8u + local_id.x;
    let output_channel = workgroup_id.y * 32u + local_id.y * 4u;
    let output_rows = params.batch * params.output_height * params.output_width;
    let k_size = params.kernel_height * params.kernel_width * params.input_channels;
    if params.output_channels > 1024u
        || (params.kernel_height == 9u && params.kernel_width == 9u && params.output_channels == 64u) {
        conv_wide(local_id, workgroup_id, output_rows, k_size);
        return;
    }
    if params.kernel_height == 1u
        && params.kernel_width == 1u
        && output_rows >= 64u {
        conv_1x1_m32(local_id, workgroup_id, output_rows);
        return;
    }
    var accum = vec4<f32>(0.0);
    var k_base = 0u;

    loop {
        if k_base >= k_size { break; }

        // Every element is assigned on every iteration, including invalid tails.
        var tile_index = local_linear;
        loop {
            if tile_index >= 128u { break; }
            let tile_row = tile_index / 16u;
            let tile_k = tile_index - tile_row * 16u;
            let global_row = row_group * 8u + tile_row;
            let global_k = k_base + tile_k;
            var value = 0.0;
            if global_row < output_rows && global_k < k_size {
                value = im2col_value(global_row, global_k);
            }
            input_tile[tile_index] = value;
            tile_index += 64u;
        }

        tile_index = local_linear;
        loop {
            if tile_index >= 512u { break; }
            let tile_k = tile_index / 32u;
            let tile_channel = tile_index - tile_k * 32u;
            let global_k = k_base + tile_k;
            let global_channel = workgroup_id.y * 32u + tile_channel;
            var value = 0.0;
            if global_k < k_size && global_channel < params.output_channels {
                value = load_weight(params.weight_offset + global_k * params.weight_k_stride + global_channel);
            }
            weight_tile[tile_index] = value;
            tile_index += 64u;
        }

        workgroupBarrier();

        let input_base = local_id.x * 16u;
        let weight_base = local_id.y * 4u;
        var partial = vec4<f32>(0.0);
        for (var tile_k = 0u; tile_k < 16u; tile_k += 1u) {
            let weight_index = tile_k * 32u + weight_base;
            let weight_value = vec4<f32>(
                weight_tile[weight_index],
                weight_tile[weight_index + 1u],
                weight_tile[weight_index + 2u],
                weight_tile[weight_index + 3u],
            );
            partial += input_tile[input_base + tile_k] * weight_value;
        }
        accum += partial;

        workgroupBarrier();
        k_base += 16u;
    }

    if output_row >= output_rows { return; }

    for (var lane = 0u; lane < 4u; lane += 1u) {
        let channel = output_channel + lane;
        if channel < params.output_channels {
            var value = accum[lane];
            if (params.flags & 1u) != 0u {
                value += load_weight(params.bias_offset + channel);
            }
            if (params.flags & 2u) != 0u {
                value += arena[params.add_offset + output_row * params.output_channel_stride + channel];
            }
            arena[params.output_offset + output_row * params.output_channel_stride + channel] =
                activate_scalar(value, params.activation);
        } else if channel < params.output_channel_stride {
            arena[params.output_offset + output_row * params.output_channel_stride + channel] = 0.0;
        }
    }
}
