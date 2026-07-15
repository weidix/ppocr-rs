// Exact F32 GEMV for single-spatial-row NHWC 1x1 convolutions. Threads compute
// independent K16 partials and the first lane combines them in original order.

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
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
    reserved4: u32,
    reserved5: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> weights: array<vec4<f32>>;

var<immediate> params: ConvParams;

// Eight output vec4 lanes, with up to 48 K16 tiles (768 input channels).
var<workgroup> partials: array<vec4<f32>, 384>;

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

fn input_scalar(channel: u32) -> f32 {
    let value = arena[(params.input_offset + channel) / 4u];
    return value[channel & 3u];
}

@compute @workgroup_size(8, 8, 1)
fn conv_single_row(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let channel = workgroup_id.y * 32u + local_id.y * 4u;
    let tile_count = (params.input_channels + 15u) / 16u;
    var tile = local_id.x;
    loop {
        if tile >= tile_count { break; }
        let k_base = tile * 16u;
        var partial = vec4<f32>(0.0);
        if channel < params.output_channel_stride {
            for (var k = 0u; k < 16u; k += 1u) {
                let global_k = k_base + k;
                if global_k < params.input_channels {
                    let weight_index = params.weight_offset
                        + global_k * params.weight_k_stride
                        + channel;
                    partial += input_scalar(global_k) * weights[weight_index / 4u];
                }
            }
        }
        partials[local_id.y * 48u + tile] = partial;
        tile += 8u;
    }
    workgroupBarrier();

    if local_id.x != 0u || channel >= params.output_channel_stride {
        return;
    }
    var result = vec4<f32>(0.0);
    for (var index = 0u; index < tile_count; index += 1u) {
        result += partials[local_id.y * 48u + index];
    }
    if (params.flags & 1u) != 0u {
        result += weights[(params.bias_offset + channel) / 4u];
    }
    if (params.flags & 2u) != 0u {
        result += arena[(params.add_offset + channel) / 4u];
    }
    arena[(params.output_offset + channel) / 4u] = activate_value(result, params.activation);
}
