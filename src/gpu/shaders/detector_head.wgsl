// Exact F32 fusion of the detector's 3x3 convolution and two 2x2 transposed
// convolutions. One workgroup produces the 4x4 output block derived from one
// neck pixel, keeping both hidden tensors in workgroup memory.

struct HeadParams {
    input_offset: u32,
    output_offset: u32,
    conv_weight_offset: u32,
    conv_bias_offset: u32,
    up_weight_offset: u32,
    up_bias_offset: u32,
    final_weight_offset: u32,
    final_bias_offset: u32,
    batch: u32,
    input_height: u32,
    input_width: u32,
    input_channels: u32,
    input_channel_stride: u32,
    hidden_channels: u32,
    output_height: u32,
    output_width: u32,
    flags: u32,
    dispatch_x: u32,
    samples_per_group: u32,
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
    reserved13: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: HeadParams;

var<workgroup> down_values: array<f32, 256>;
var<workgroup> up_values: array<f32, 1024>;

@compute @workgroup_size(256, 1, 1)
fn detector_head(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let group_index = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let input_plane = params.input_height * params.input_width;
    let input_rows = params.batch * input_plane;
    let row_base = group_index * params.samples_per_group;
    if row_base >= input_rows {
        return;
    }
    let sample = lane / params.hidden_channels;
    let hidden_channel = lane - sample * params.hidden_channels;
    let input_row = row_base + sample;
    let batch_index = input_row / input_plane;
    let spatial = input_row - batch_index * input_plane;
    let input_y = spatial / params.input_width;
    let input_x = spatial - input_y * params.input_width;
    let valid_sample = sample < params.samples_per_group && input_row < input_rows;

    if sample < params.samples_per_group && hidden_channel < params.hidden_channels {
        var value = 0.0;
        if valid_sample {
            if (params.flags & 1u) != 0u {
                value = weights[params.conv_bias_offset + hidden_channel];
            }
            for (var kernel_y = 0u; kernel_y < 3u; kernel_y += 1u) {
                let source_y = i32(input_y + kernel_y) - 1;
                if source_y < 0 || source_y >= i32(params.input_height) {
                    continue;
                }
                for (var kernel_x = 0u; kernel_x < 3u; kernel_x += 1u) {
                    let source_x = i32(input_x + kernel_x) - 1;
                    if source_x < 0 || source_x >= i32(params.input_width) {
                        continue;
                    }
                    let source_base = params.input_offset
                        + (((batch_index * params.input_height + u32(source_y)) * params.input_width
                        + u32(source_x)) * params.input_channel_stride);
                    let weight_base = params.conv_weight_offset
                        + (kernel_y * 3u + kernel_x) * params.input_channels * params.hidden_channels
                        + hidden_channel;
                    for (var channel = 0u; channel < params.input_channels; channel += 1u) {
                        value += arena[source_base + channel]
                            * weights[weight_base + channel * params.hidden_channels];
                    }
                }
            }
            value = max(value, 0.0);
        }
        down_values[sample * params.hidden_channels + hidden_channel] = value;
    }
    workgroupBarrier();

    let up_count = params.samples_per_group * params.hidden_channels * 4u;
    for (var index = lane; index < up_count; index += 256u) {
        let sample_phase = index / params.hidden_channels;
        let source_sample = sample_phase / 4u;
        let phase = sample_phase - source_sample * 4u;
        let output_channel = index - sample_phase * params.hidden_channels;
        var value = 0.0;
        if (params.flags & 2u) != 0u {
            value = weights[params.up_bias_offset + output_channel];
        }
        let weight_base = params.up_weight_offset
            + phase * params.hidden_channels * params.hidden_channels
            + output_channel;
        for (var channel = 0u; channel < params.hidden_channels; channel += 1u) {
            value += down_values[source_sample * params.hidden_channels + channel]
                * weights[weight_base + channel * params.hidden_channels];
        }
        up_values[(source_sample * 4u + phase) * params.hidden_channels + output_channel] = max(value, 0.0);
    }
    workgroupBarrier();

    if lane < params.samples_per_group * 16u {
        let source_sample = lane / 16u;
        let output_in_sample = lane - source_sample * 16u;
        let source_row = row_base + source_sample;
        if source_row >= input_rows {
            return;
        }
        let source_batch = source_row / input_plane;
        let source_spatial = source_row - source_batch * input_plane;
        let source_y = source_spatial / params.input_width;
        let source_x = source_spatial - source_y * params.input_width;
        let block_y = output_in_sample / 4u;
        let block_x = output_in_sample - block_y * 4u;
        let up_phase = (block_y / 2u) * 2u + block_x / 2u;
        let final_phase = (block_y & 1u) * 2u + (block_x & 1u);
        var value = 0.0;
        if (params.flags & 4u) != 0u {
            value = weights[params.final_bias_offset];
        }
        let weight_base = params.final_weight_offset
            + final_phase * params.hidden_channels * 4u;
        for (var channel = 0u; channel < params.hidden_channels; channel += 1u) {
            value += up_values[(source_sample * 4u + up_phase) * params.hidden_channels + channel]
                * weights[weight_base + channel * 4u];
        }
        let output_y = source_y * 4u + block_y;
        let output_x = source_x * 4u + block_x;
        let output_index = params.output_offset
            + (((source_batch * params.output_height + output_y) * params.output_width + output_x)
            * 4u);
        arena[output_index] = 1.0 / (1.0 + exp(-value));
    }
}
