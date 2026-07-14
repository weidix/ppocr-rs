// GPU runtime: stable row-wise layer normalization over the logical NHWC channel axis.

struct LayerNormParams {
    input_offset: u32,
    output_offset: u32,
    weight_offset: u32,
    bias_offset: u32,
    rows: u32,
    channels: u32,
    input_stride: u32,
    output_stride: u32,
    epsilon_bits: u32,
    dispatch_x: u32,
    reserved0: u32,
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
    reserved18: u32,
    reserved19: u32,
    reserved20: u32,
    reserved21: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: LayerNormParams;

var<workgroup> partial_mean: array<f32, 256>;
var<workgroup> partial_m2: array<f32, 256>;
var<workgroup> partial_count: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn layer_norm(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    if row >= params.rows { return; }

    let input_base = params.input_offset + row * params.input_stride;
    var count = 0u;
    var mean = 0.0;
    var m2 = 0.0;
    for (var channel = local_id.x; channel < params.channels; channel += 256u) {
        let value = arena[input_base + channel];
        count += 1u;
        let delta = value - mean;
        mean += delta / f32(count);
        m2 += delta * (value - mean);
    }
    partial_mean[local_id.x] = mean;
    partial_m2[local_id.x] = m2;
    partial_count[local_id.x] = count;
    workgroupBarrier();

    var reduction_stride = 128u;
    loop {
        if reduction_stride == 0u { break; }
        if local_id.x < reduction_stride {
            let right_count = partial_count[local_id.x + reduction_stride];
            if right_count != 0u {
                let left_count = partial_count[local_id.x];
                if left_count == 0u {
                    partial_mean[local_id.x] = partial_mean[local_id.x + reduction_stride];
                    partial_m2[local_id.x] = partial_m2[local_id.x + reduction_stride];
                    partial_count[local_id.x] = right_count;
                } else {
                    let combined_count = left_count + right_count;
                    let delta = partial_mean[local_id.x + reduction_stride] - partial_mean[local_id.x];
                    partial_mean[local_id.x] += delta * f32(right_count) / f32(combined_count);
                    partial_m2[local_id.x] += partial_m2[local_id.x + reduction_stride]
                        + delta * delta * f32(left_count) * f32(right_count) / f32(combined_count);
                    partial_count[local_id.x] = combined_count;
                }
            }
        }
        workgroupBarrier();
        reduction_stride /= 2u;
    }

    let variance = max(partial_m2[0] / f32(params.channels), 0.0);
    let inverse_std = inverseSqrt(variance + bitcast<f32>(params.epsilon_bits));
    let output_base = params.output_offset + row * params.output_stride;
    for (var channel = local_id.x; channel < params.channels; channel += 256u) {
        let normalized = (arena[input_base + channel] - partial_mean[0]) * inverse_std;
        arena[output_base + channel] = normalized * weights[params.weight_offset + channel]
            + weights[params.bias_offset + channel];
    }
    for (var channel = params.channels + local_id.x; channel < params.output_stride; channel += 256u) {
        arena[output_base + channel] = 0.0;
    }
}
