// GPU runtime: NHWC global and windowed pooling.

struct PoolParams {
    input_offset: u32,
    output_offset: u32,
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
    activation: u32,
    flags: u32,
    dispatch_x: u32,
    channel_groups: u32,
    reserved2: u32,
    reserved3: u32,
    reserved4: u32,
    reserved5: u32,
    reserved6: u32,
    reserved7: u32,
    reserved8: u32,
    reserved9: u32,
    reserved10: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: PoolParams;

var<workgroup> mean_partial: array<vec4<f32>, 256>;

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

fn window_pool(output_row: u32, channel_base: u32, use_max: bool) -> vec4<f32> {
    let output_plane = params.output_height * params.output_width;
    let batch_index = output_row / output_plane;
    let output_spatial = output_row - batch_index * output_plane;
    let output_y = output_spatial / params.output_width;
    let output_x = output_spatial - output_y * params.output_width;
    var result = select(vec4<f32>(0.0), vec4<f32>(-3.402823466e38), use_max);
    var valid_count = 0u;

    for (var kernel_y = 0u; kernel_y < params.kernel_height; kernel_y += 1u) {
        let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) - i32(params.pad_top);
        if input_y < 0 || input_y >= i32(params.input_height) { continue; }
        for (var kernel_x = 0u; kernel_x < params.kernel_width; kernel_x += 1u) {
            let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) - i32(params.pad_left);
            if input_x < 0 || input_x >= i32(params.input_width) { continue; }
            let input_base = params.input_offset
                + (((batch_index * params.input_height + u32(input_y)) * params.input_width + u32(input_x))
                * params.input_channel_stride)
                + channel_base;
            for (var lane = 0u; lane < 4u; lane += 1u) {
                if channel_base + lane < params.input_channels && channel_base + lane < params.output_channels {
                    if use_max {
                        result[lane] = max(result[lane], arena[input_base + lane]);
                    } else {
                        result[lane] += arena[input_base + lane];
                    }
                }
            }
            valid_count += 1u;
        }
    }

    if !use_max {
        var divisor = valid_count;
        if (params.flags & 1u) != 0u {
            divisor = params.kernel_height * params.kernel_width;
        }
        if divisor > 0u {
            result /= f32(divisor);
        }
    } else if valid_count == 0u {
        result = vec4<f32>(0.0);
    }
    return result;
}

fn write_window(output_row: u32, channel_base: u32, value: vec4<f32>) {
    let output_base = params.output_offset + output_row * params.output_channel_stride + channel_base;
    for (var lane = 0u; lane < 4u; lane += 1u) {
        let channel = channel_base + lane;
        if channel < params.output_channels {
            arena[output_base + lane] = activate_scalar(value[lane], params.activation);
        } else {
            arena[output_base + lane] = 0.0;
        }
    }
}

fn row_index(local_id: vec3<u32>, workgroup_id: vec3<u32>) -> u32 {
    return (workgroup_id.z * params.dispatch_x + workgroup_id.x) * 8u + local_id.x;
}

@compute @workgroup_size(8, 8, 1)
fn pool_max(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let output_row = row_index(local_id, workgroup_id);
    let channel_base = (workgroup_id.y * 8u + local_id.y) * 4u;
    if output_row >= params.batch * params.output_height * params.output_width
        || channel_base >= params.output_channel_stride {
        return;
    }
    write_window(output_row, channel_base, window_pool(output_row, channel_base, true));
}

@compute @workgroup_size(8, 8, 1)
fn pool_avg(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let output_row = row_index(local_id, workgroup_id);
    let channel_base = (workgroup_id.y * 8u + local_id.y) * 4u;
    if output_row >= params.batch * params.output_height * params.output_width
        || channel_base >= params.output_channel_stride {
        return;
    }
    write_window(output_row, channel_base, window_pool(output_row, channel_base, false));
}

// flags bit 1 selects max pooling; clear selects average pooling.
@compute @workgroup_size(8, 8, 1)
fn pool(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let output_row = row_index(local_id, workgroup_id);
    let channel_base = (workgroup_id.y * 8u + local_id.y) * 4u;
    if output_row >= params.batch * params.output_height * params.output_width
        || channel_base >= params.output_channel_stride {
        return;
    }
    write_window(
        output_row,
        channel_base,
        window_pool(output_row, channel_base, (params.flags & 2u) != 0u),
    );
}

@compute @workgroup_size(256, 1, 1)
fn global_mean(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let linear_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let channel_base = (linear_group % params.channel_groups) * 4u;
    let batch_index = linear_group / params.channel_groups;
    let spatial_size = params.input_height * params.input_width;
    var partial = vec4<f32>(0.0);

    if batch_index < params.batch && channel_base < params.output_channel_stride {
        for (var spatial = local_id.x; spatial < spatial_size; spatial += 256u) {
            let input_base = params.input_offset
                + (batch_index * spatial_size + spatial) * params.input_channel_stride
                + channel_base;
            for (var lane = 0u; lane < 4u; lane += 1u) {
                if channel_base + lane < params.input_channels && channel_base + lane < params.output_channels {
                    partial[lane] += arena[input_base + lane];
                }
            }
        }
    }

    // Every invocation initializes its slot before the first reduction barrier.
    mean_partial[local_id.x] = partial;
    workgroupBarrier();

    var reduction_stride = 128u;
    loop {
        if reduction_stride == 0u { break; }
        if local_id.x < reduction_stride {
            mean_partial[local_id.x] += mean_partial[local_id.x + reduction_stride];
        }
        workgroupBarrier();
        reduction_stride /= 2u;
    }

    if local_id.x == 0u && batch_index < params.batch && channel_base < params.output_channel_stride {
        let output_base = params.output_offset + batch_index * params.output_channel_stride + channel_base;
        for (var lane = 0u; lane < 4u; lane += 1u) {
            let channel = channel_base + lane;
            if channel < params.output_channels && spatial_size > 0u {
                arena[output_base + lane] = activate_scalar(mean_partial[0][lane] / f32(spatial_size), params.activation);
            } else {
                arena[output_base + lane] = 0.0;
            }
        }
    }
}
