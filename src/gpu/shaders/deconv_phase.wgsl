// GPU runtime: NHWC 2x2 stride-2 transposed convolution as four independent 1x1 GEMMs.
// Each workgroup computes 32 input rows and 32 output channels for one phase.

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

fn load_weight(index: u32) -> f32 {
    return weights[index];
}

var<immediate> params: DeconvParams;

var<workgroup> input_tile: array<f32, 544>;
var<workgroup> weight_tile: array<vec4<f32>, 128>;

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

fn output_row(input_row: u32, phase: u32) -> u32 {
    let input_plane = params.input_height * params.input_width;
    let output_plane = params.output_height * params.output_width;
    let batch_index = input_row / input_plane;
    let input_spatial = input_row - batch_index * input_plane;
    let input_y = input_spatial / params.input_width;
    let input_x = input_spatial - input_y * params.input_width;
    let output_y = input_y * 2u + phase / 2u;
    let output_x = input_x * 2u + (phase & 1u);
    return batch_index * output_plane + output_y * params.output_width + output_x;
}

fn write_output(row: u32, channel: u32, value: f32) {
    var result = value;
    if (params.flags & 1u) != 0u {
        result += load_weight(params.bias_offset + channel);
    }
    if (params.flags & 2u) != 0u {
        result += arena[params.add_offset + row * params.output_channel_stride + channel];
    }
    arena[params.output_offset + row * params.output_channel_stride + channel] =
        activate_scalar(result, params.activation);
}

@compute @workgroup_size(16, 8, 1)
fn deconv_phase(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let logical_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let phase = logical_group & 3u;
    let row_base = (logical_group >> 2u) * 32u;
    let input_rows = params.batch * params.input_height * params.input_width;
    if row_base >= input_rows {
        return;
    }

    let local_linear = local_id.y * 16u + local_id.x;
    let output_channel = workgroup_id.y * 32u + local_id.y * 4u;
    let row0 = row_base + local_id.x;
    let row1 = row0 + 16u;
    var accum0 = vec4<f32>(0.0);
    var accum1 = vec4<f32>(0.0);

    for (var k_base = 0u; k_base < params.input_channels; k_base += 16u) {
        var tile_index = local_linear;
        loop {
            if tile_index >= 512u { break; }
            let tile_row = tile_index / 16u;
            let tile_k = tile_index - tile_row * 16u;
            let input_row = row_base + tile_row;
            var value = 0.0;
            if input_row < input_rows {
                value = arena[params.input_offset
                    + input_row * params.input_channel_stride
                    + k_base + tile_k];
            }
            input_tile[tile_row * 17u + tile_k] = value;
            tile_index += 128u;
        }

        let tile_k = local_linear / 8u;
        let channel4 = local_linear - tile_k * 8u;
        let source = params.weight_offset
            + (phase * params.input_channels + k_base + tile_k) * params.weight_k_stride
            + workgroup_id.y * 32u
            + channel4 * 4u;
        weight_tile[local_linear] = vec4<f32>(
            load_weight(source),
            load_weight(source + 1u),
            load_weight(source + 2u),
            load_weight(source + 3u),
        );

        workgroupBarrier();

        let input_base0 = local_id.x * 17u;
        let input_base1 = (local_id.x + 16u) * 17u;
        for (var k = 0u; k < 16u; k += 1u) {
            let weight_value = weight_tile[k * 8u + local_id.y];
            accum0 += input_tile[input_base0 + k] * weight_value;
            accum1 += input_tile[input_base1 + k] * weight_value;
        }

        workgroupBarrier();
    }

    if row0 < input_rows {
        let target_row = output_row(row0, phase);
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(target_row, output_channel + lane, accum0[lane]);
        }
    }
    if row1 < input_rows {
        let target_row = output_row(row1, phase);
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(target_row, output_channel + lane, accum1[lane]);
        }
    }
}
