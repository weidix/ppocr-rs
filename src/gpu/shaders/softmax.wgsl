// GPU runtime: stable row-wise softmax. An optional same-shape additive mask is read from
// arena when flags bit 0 is set. scale_bits contains bitcast<u32>(scale).

struct SoftmaxParams {
    input_offset: u32,
    output_offset: u32,
    add_offset: u32,
    rows: u32,
    logical_size: u32,
    input_stride: u32,
    output_stride: u32,
    scale_bits: u32,
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
    reserved13: u32,
    reserved14: u32,
    reserved15: u32,
    reserved16: u32,
    reserved17: u32,
    reserved18: u32,
    reserved19: u32,
    reserved20: u32,
    reserved21: u32,
    reserved22: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: SoftmaxParams;

var<workgroup> reduction: array<f32, 256>;

fn input_value(row: u32, column: u32) -> f32 {
    var value = arena[params.input_offset + row * params.input_stride + column];
    if (params.flags & 1u) != 0u {
        value += arena[params.add_offset + row * params.input_stride + column];
    }
    return value * bitcast<f32>(params.scale_bits);
}

@compute @workgroup_size(256, 1, 1)
fn softmax(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    if row >= params.rows { return; }

    if params.logical_size == 0u {
        for (var column = local_id.x; column < params.output_stride; column += 256u) {
            arena[params.output_offset + row * params.output_stride + column] = 0.0;
        }
        return;
    }

    var local_max = -3.402823466e38;
    for (var column = local_id.x; column < params.logical_size; column += 256u) {
        local_max = max(local_max, input_value(row, column));
    }
    reduction[local_id.x] = local_max;
    workgroupBarrier();

    var reduction_stride = 128u;
    loop {
        if reduction_stride == 0u { break; }
        if local_id.x < reduction_stride {
            reduction[local_id.x] = max(reduction[local_id.x], reduction[local_id.x + reduction_stride]);
        }
        workgroupBarrier();
        reduction_stride /= 2u;
    }
    let row_max = reduction[0];
    workgroupBarrier();

    var local_sum = 0.0;
    for (var column = local_id.x; column < params.logical_size; column += 256u) {
        local_sum += exp(input_value(row, column) - row_max);
    }
    reduction[local_id.x] = local_sum;
    workgroupBarrier();

    reduction_stride = 128u;
    loop {
        if reduction_stride == 0u { break; }
        if local_id.x < reduction_stride {
            reduction[local_id.x] += reduction[local_id.x + reduction_stride];
        }
        workgroupBarrier();
        reduction_stride /= 2u;
    }
    let inverse_sum = 1.0 / reduction[0];

    for (var column = local_id.x; column < params.logical_size; column += 256u) {
        arena[params.output_offset + row * params.output_stride + column] =
            exp(input_value(row, column) - row_max) * inverse_sum;
    }
    for (var column = params.logical_size + local_id.x; column < params.output_stride; column += 256u) {
        arena[params.output_offset + row * params.output_stride + column] = 0.0;
    }
}
