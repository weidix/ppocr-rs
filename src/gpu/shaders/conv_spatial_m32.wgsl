// GPU runtime: NHWC ungrouped spatial convolution using an M32xN32xK32 tile.
// This targets medium-detector intraclass convolutions with 32 channels.

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

// The padded K stride avoids same-bank reads across output rows.
var<workgroup> input_tile: array<f32, 1056>;
var<workgroup> weight_tile: array<vec4<f32>, 256>;

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
    let output_index = params.output_offset + output_row * params.output_channel_stride + channel;
    if channel < params.output_channels {
        var result = value;
        if (params.flags & 1u) != 0u {
            result += weights[params.bias_offset + channel];
        }
        if (params.flags & 2u) != 0u {
            result += arena[params.add_offset + output_row * params.output_channel_stride + channel];
        }
        arena[output_index] = activate_scalar(result, params.activation);
    } else if channel < params.output_channel_stride {
        arena[output_index] = 0.0;
    }
}

@compute @workgroup_size(8, 8, 1)
fn conv_spatial_m32(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.reserved0 + workgroup_id.x;
    let tiles_per_row = params.reserved1;
    let row_tiles = params.batch * params.output_height * tiles_per_row;
    if row_group >= row_tiles || workgroup_id.y >= (params.output_channels + 31u) / 32u {
        return;
    }

    let local_linear = local_id.y * 8u + local_id.x;
    let tile_x = row_group % tiles_per_row;
    let output_yb = row_group / tiles_per_row;
    let batch_index = output_yb / params.output_height;
    let output_y = output_yb - batch_index * params.output_height;
    let output_x0 = tile_x * 32u + local_id.x;
    let output_x1 = output_x0 + 8u;
    let output_x2 = output_x0 + 16u;
    let output_x3 = output_x0 + 24u;
    let output_row0 = (batch_index * params.output_height + output_y)
        * params.output_width + output_x0;
    let output_row1 = output_row0 + 8u;
    let output_row2 = output_row0 + 16u;
    let output_row3 = output_row0 + 24u;
    let output_channel = workgroup_id.y * 32u + local_id.y * 4u;
    var accum0 = vec4<f32>(0.0);
    var accum1 = vec4<f32>(0.0);
    var accum2 = vec4<f32>(0.0);
    var accum3 = vec4<f32>(0.0);
    let k_size = params.kernel_height * params.kernel_width * params.input_channels;
    var k_base = 0u;

    loop {
        if k_base >= k_size { break; }

        let tile_k = local_linear & 31u;
        let global_k = k_base + tile_k;
        let input_channel = global_k % params.input_channels;
        let kernel_spatial = global_k / params.input_channels;
        let kernel_y = kernel_spatial / params.kernel_width;
        let kernel_x = kernel_spatial - kernel_y * params.kernel_width;
        let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y)
            - i32(params.pad_top);
        let valid_y = input_y >= 0 && input_y < i32(params.input_height);
        var input_row_base = 0u;
        if valid_y {
            input_row_base = params.input_offset
                + ((batch_index * params.input_height + u32(input_y)) * params.input_width
                * params.input_channel_stride)
                + input_channel;
        }
        var tile_row = local_linear / 32u;
        loop {
            if tile_row >= 32u { break; }
            let output_x = tile_x * 32u + tile_row;
            var value = 0.0;
            if output_x < params.output_width && global_k < k_size && valid_y {
                let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x)
                    - i32(params.pad_left);
                if input_x >= 0 && input_x < i32(params.input_width) {
                    value = arena[input_row_base
                        + u32(input_x) * params.input_channel_stride];
                }
            }
            input_tile[tile_row * 33u + tile_k] = value;
            tile_row += 2u;
        }

        var weight_index = local_linear;
        loop {
            if weight_index >= 256u { break; }
            let tile_k = weight_index / 8u;
            let channel4 = weight_index - tile_k * 8u;
            let global_k = k_base + tile_k;
            let global_channel = workgroup_id.y * 32u + channel4 * 4u;
            let source = params.weight_offset
                + global_k * params.weight_k_stride
                + global_channel;
            var value = vec4<f32>(0.0);
            if global_k < k_size && global_channel < params.output_channels {
                value = vec4<f32>(
                    weights[source],
                    weights[source + 1u],
                    weights[source + 2u],
                    weights[source + 3u],
                );
            }
            weight_tile[weight_index] = value;
            weight_index += 64u;
        }

        workgroupBarrier();

        let weight_lane = local_id.y;
        let input_base0 = local_id.x * 33u;
        let input_base1 = (local_id.x + 8u) * 33u;
        let input_base2 = (local_id.x + 16u) * 33u;
        let input_base3 = (local_id.x + 24u) * 33u;
        for (var k = 0u; k < 32u; k += 1u) {
            let weight_value = weight_tile[k * 8u + weight_lane];
            accum0 += input_tile[input_base0 + k] * weight_value;
            accum1 += input_tile[input_base1 + k] * weight_value;
            accum2 += input_tile[input_base2 + k] * weight_value;
            accum3 += input_tile[input_base3 + k] * weight_value;
        }

        workgroupBarrier();
        k_base += 32u;
    }

    if output_x0 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row0, output_channel + lane, accum0[lane]);
        }
    }
    if output_x1 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row1, output_channel + lane, accum1[lane]);
        }
    }
    if output_x2 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row2, output_channel + lane, accum2[lane]);
        }
    }
    if output_x3 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row3, output_channel + lane, accum3[lane]);
        }
    }
}
