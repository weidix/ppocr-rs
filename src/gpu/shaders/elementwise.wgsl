// GPU runtime: NHWC elementwise, resize, and channel-concatenation kernels.

struct ElementwiseParams {
    src0_offset: u32,
    src1_offset: u32,
    dst_offset: u32,
    batch: u32,
    src0_height: u32,
    src0_width: u32,
    src0_channels: u32,
    src0_channel_stride: u32,
    src1_batch: u32,
    src1_height: u32,
    src1_width: u32,
    src1_channels: u32,
    src1_channel_stride: u32,
    dst_height: u32,
    dst_width: u32,
    dst_channels: u32,
    dst_channel_stride: u32,
    activation: u32,
    flags: u32,
    dispatch_x: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
    reserved4: u32,
    reserved5: u32,
    reserved6: u32,
    reserved7: u32,
    reserved8: u32,
    reserved9: u32,
    reserved10: u32,
    reserved11: u32,
    reserved12: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: ElementwiseParams;

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

fn decode_block(linear: u32) -> vec4<u32> {
    let blocks_per_pixel = params.dst_channel_stride / 4u;
    let pixel = linear / blocks_per_pixel;
    let channel = (linear - pixel * blocks_per_pixel) * 4u;
    let batch_plane = params.dst_height * params.dst_width;
    let batch_index = pixel / batch_plane;
    let spatial = pixel - batch_index * batch_plane;
    let y = spatial / params.dst_width;
    let x = spatial - y * params.dst_width;
    return vec4<u32>(batch_index, y, x, channel);
}

fn block_count() -> u32 {
    return params.batch * params.dst_height * params.dst_width * (params.dst_channel_stride / 4u);
}

fn linear_index(local_id: vec3<u32>, workgroup_id: vec3<u32>) -> u32 {
    let workgroup = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    return workgroup * 256u + local_id.x;
}

@compute @workgroup_size(256, 1, 1)
fn add_same(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = linear_index(local_id, workgroup_id);
    if linear >= block_count() { return; }
    let coord = decode_block(linear);
    let src0_base = params.src0_offset
        + (((coord.x * params.src0_height + coord.y) * params.src0_width + coord.z)
        * params.src0_channel_stride)
        + coord.w;
    let src1_base = params.src1_offset
        + (((coord.x * params.src1_height + coord.y) * params.src1_width + coord.z)
        * params.src1_channel_stride)
        + coord.w;
    let dst_base = params.dst_offset
        + (((coord.x * params.dst_height + coord.y) * params.dst_width + coord.z)
        * params.dst_channel_stride)
        + coord.w;

    for (var lane = 0u; lane < 4u; lane += 1u) {
        let channel = coord.w + lane;
        if channel < params.dst_channels && channel < params.src0_channels && channel < params.src1_channels {
            arena[dst_base + lane] = activate_scalar(arena[src0_base + lane] + arena[src1_base + lane], params.activation);
        } else {
            arena[dst_base + lane] = 0.0;
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn mul_broadcast(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = linear_index(local_id, workgroup_id);
    if linear >= block_count() { return; }
    let coord = decode_block(linear);
    let rhs_batch = select(0u, coord.x, params.src1_batch > 1u);
    let rhs_y = select(0u, coord.y, params.src1_height > 1u);
    let rhs_x = select(0u, coord.z, params.src1_width > 1u);
    let src0_base = params.src0_offset
        + (((coord.x * params.src0_height + coord.y) * params.src0_width + coord.z)
        * params.src0_channel_stride)
        + coord.w;
    let dst_base = params.dst_offset
        + (((coord.x * params.dst_height + coord.y) * params.dst_width + coord.z)
        * params.dst_channel_stride)
        + coord.w;

    for (var lane = 0u; lane < 4u; lane += 1u) {
        let channel = coord.w + lane;
        if channel < params.dst_channels && channel < params.src0_channels {
            let rhs_channel = select(channel, 0u, params.src1_channels == 1u);
            var rhs = 0.0;
            if rhs_channel < params.src1_channels {
                let rhs_index = params.src1_offset
                    + (((rhs_batch * params.src1_height + rhs_y) * params.src1_width + rhs_x)
                    * params.src1_channel_stride)
                    + rhs_channel;
                rhs = arena[rhs_index];
            }
            arena[dst_base + lane] = activate_scalar(arena[src0_base + lane] * rhs, params.activation);
        } else {
            arena[dst_base + lane] = 0.0;
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn resize_nearest(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = linear_index(local_id, workgroup_id);
    if linear >= block_count() { return; }
    let coord = decode_block(linear);
    let src_y = min((coord.y * params.src0_height) / params.dst_height, params.src0_height - 1u);
    let src_x = min((coord.z * params.src0_width) / params.dst_width, params.src0_width - 1u);
    let src_base = params.src0_offset
        + (((coord.x * params.src0_height + src_y) * params.src0_width + src_x)
        * params.src0_channel_stride)
        + coord.w;
    let dst_base = params.dst_offset
        + (((coord.x * params.dst_height + coord.y) * params.dst_width + coord.z)
        * params.dst_channel_stride)
        + coord.w;

    for (var lane = 0u; lane < 4u; lane += 1u) {
        let channel = coord.w + lane;
        if channel < params.dst_channels && channel < params.src0_channels {
            arena[dst_base + lane] = activate_scalar(arena[src_base + lane], params.activation);
        } else {
            arena[dst_base + lane] = 0.0;
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn concat2(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear = linear_index(local_id, workgroup_id);
    if linear >= block_count() { return; }
    let coord = decode_block(linear);
    let dst_base = params.dst_offset
        + (((coord.x * params.dst_height + coord.y) * params.dst_width + coord.z)
        * params.dst_channel_stride)
        + coord.w;

    for (var lane = 0u; lane < 4u; lane += 1u) {
        let channel = coord.w + lane;
        var value = 0.0;
        if channel < params.src0_channels {
            let src_index = params.src0_offset
                + (((coord.x * params.src0_height + coord.y) * params.src0_width + coord.z)
                * params.src0_channel_stride)
                + channel;
            value = arena[src_index];
        } else if channel < params.dst_channels {
            let src_channel = channel - params.src0_channels;
            if src_channel < params.src1_channels {
                let src_index = params.src1_offset
                    + (((coord.x * params.src1_height + coord.y) * params.src1_width + coord.z)
                    * params.src1_channel_stride)
                    + src_channel;
                value = arena[src_index];
            }
        }
        arena[dst_base + lane] = activate_scalar(value, params.activation);
    }
}
