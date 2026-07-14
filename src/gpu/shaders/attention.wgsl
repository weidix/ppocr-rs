// GPU runtime: fused scaled QK^T plus stable softmax, followed by probability-times-V.
// QKV channels are contiguous Q | K | V blocks, each split into heads.

struct AttentionParams {
    qkv_offset: u32,
    scores_offset: u32,
    output_offset: u32,
    batch: u32,
    sequence: u32,
    hidden_channels: u32,
    qkv_stride: u32,
    scores_stride: u32,
    output_stride: u32,
    num_heads: u32,
    head_dim: u32,
    scale_bits: u32,
    score_rows: u32,
    reserved0: u32,
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
    reserved13: u32,
    reserved14: u32,
    reserved15: u32,
    reserved16: u32,
    reserved17: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: AttentionParams;

var<workgroup> shared_values: array<f32, 64>;

fn linear_group(workgroup_id: vec3<u32>) -> u32 {
    return workgroup_id.z * params.dispatch_x + workgroup_id.x;
}

fn decode_score_row(row: u32) -> vec3<u32> {
    let query = row % params.sequence;
    let batch_head = row / params.sequence;
    let head = batch_head % params.num_heads;
    let batch_index = batch_head / params.num_heads;
    return vec3<u32>(batch_index, head, query);
}

@compute @workgroup_size(64, 1, 1)
fn attention_scores(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = linear_group(workgroup_id);
    if row >= params.score_rows { return; }

    let coord = decode_score_row(row);
    let head_offset = coord.y * params.head_dim;
    let query_base = params.qkv_offset
        + (coord.x * params.sequence + coord.z) * params.qkv_stride
        + head_offset;
    let score_base = params.scores_offset + row * params.scores_stride;
    let scale = bitcast<f32>(params.scale_bits);
    var local_max = -3.402823466e38;

    for (var key_base = 0u; key_base < params.sequence; key_base += 64u) {
        let key = key_base + local_id.x;
        var dot = 0.0;
        for (var dim_base = 0u; dim_base < params.head_dim; dim_base += 64u) {
            let dim = dim_base + local_id.x;
            if dim < params.head_dim {
                shared_values[local_id.x] = arena[query_base + dim];
            } else {
                shared_values[local_id.x] = 0.0;
            }
            workgroupBarrier();
            if key < params.sequence {
                let key_base_offset = params.qkv_offset
                    + (coord.x * params.sequence + key) * params.qkv_stride
                    + params.hidden_channels
                    + head_offset;
                let tile_size = min(64u, params.head_dim - dim_base);
                for (var tile_dim = 0u; tile_dim < tile_size; tile_dim += 1u) {
                    dot += shared_values[tile_dim] * arena[key_base_offset + dim_base + tile_dim];
                }
            }
            workgroupBarrier();
        }
        if key < params.sequence {
            let score = dot * scale;
            arena[score_base + key] = score;
            local_max = max(local_max, score);
        }
    }

    shared_values[local_id.x] = local_max;
    workgroupBarrier();
    var reduction_stride = 32u;
    loop {
        if reduction_stride == 0u { break; }
        if local_id.x < reduction_stride {
            shared_values[local_id.x] = max(
                shared_values[local_id.x],
                shared_values[local_id.x + reduction_stride],
            );
        }
        workgroupBarrier();
        reduction_stride /= 2u;
    }
    let row_max = shared_values[0];
    workgroupBarrier();

    var local_sum = 0.0;
    for (var key = local_id.x; key < params.sequence; key += 64u) {
        local_sum += exp(arena[score_base + key] - row_max);
    }
    shared_values[local_id.x] = local_sum;
    workgroupBarrier();
    reduction_stride = 32u;
    loop {
        if reduction_stride == 0u { break; }
        if local_id.x < reduction_stride {
            shared_values[local_id.x] += shared_values[local_id.x + reduction_stride];
        }
        workgroupBarrier();
        reduction_stride /= 2u;
    }
    let inverse_sum = 1.0 / shared_values[0];

    for (var key = local_id.x; key < params.sequence; key += 64u) {
        arena[score_base + key] = exp(arena[score_base + key] - row_max) * inverse_sum;
    }
    for (var key = params.sequence + local_id.x; key < params.scores_stride; key += 64u) {
        arena[score_base + key] = 0.0;
    }
}

@compute @workgroup_size(64, 1, 1)
fn attention_context(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = linear_group(workgroup_id);
    if row >= params.score_rows { return; }

    let coord = decode_score_row(row);
    let head_offset = coord.y * params.head_dim;
    let score_base = params.scores_offset + row * params.scores_stride;
    let output_base = params.output_offset
        + (coord.x * params.sequence + coord.z) * params.output_stride
        + head_offset;

    for (var channel_base = 0u; channel_base < params.head_dim; channel_base += 64u) {
        let channel = channel_base + local_id.x;
        var value = 0.0;
        for (var key_base = 0u; key_base < params.sequence; key_base += 64u) {
            let key = key_base + local_id.x;
            if key < params.sequence {
                shared_values[local_id.x] = arena[score_base + key];
            } else {
                shared_values[local_id.x] = 0.0;
            }
            workgroupBarrier();
            if channel < params.head_dim {
                let tile_size = min(64u, params.sequence - key_base);
                for (var tile_key = 0u; tile_key < tile_size; tile_key += 1u) {
                    let value_base = params.qkv_offset
                        + (coord.x * params.sequence + key_base + tile_key) * params.qkv_stride
                        + params.hidden_channels * 2u
                        + head_offset;
                    value += shared_values[tile_key] * arena[value_base + channel];
                }
            }
            workgroupBarrier();
        }
        if channel < params.head_dim {
            arena[output_base + channel] = value;
        }
    }

    if coord.y + 1u == params.num_heads {
        let padded_base = params.output_offset
            + (coord.x * params.sequence + coord.z) * params.output_stride;
        for (var channel = params.hidden_channels + local_id.x;
            channel < params.output_stride;
            channel += 64u) {
            arena[padded_base + channel] = 0.0;
        }
    }
}
