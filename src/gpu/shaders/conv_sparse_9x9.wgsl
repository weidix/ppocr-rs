// Exact F32 9x9 convolution for weights with structured input-channel
// sparsity. Active channels are traversed in their original ascending order,
// so omitted multiplications are exclusively multiplications by exact zero.

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
    tiles_per_row: u32,
    sparse_channels_offset: u32,
    sparse_channel_count: u32,
    reserved2: u32,
    reserved3: u32,
    reserved4: u32,
}

@group(0) @binding(0)
var<storage, read_write> arena: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

var<immediate> params: ConvParams;

var<workgroup> input_tile: array<f32, 1056>;
var<workgroup> weight_tile: array<vec4<f32>, 512>;

fn write_output(output_row: u32, channel: u32, value: f32) {
    var result = value;
    if (params.flags & 1u) != 0u {
        result += weights[params.bias_offset + channel];
    }
    if (params.flags & 2u) != 0u {
        result += arena[params.add_offset + output_row * params.output_channel_stride + channel];
    }
    arena[params.output_offset + output_row * params.output_channel_stride + channel] = result;
}

@compute @workgroup_size(16, 8, 1)
fn conv_sparse_9x9(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_group = workgroup_id.z * params.dispatch_x + workgroup_id.x;
    let row_tiles = params.batch * params.output_height * params.tiles_per_row;
    if row_group >= row_tiles {
        return;
    }
    let tile_x = row_group % params.tiles_per_row;
    let output_yb = row_group / params.tiles_per_row;
    let batch_index = output_yb / params.output_height;
    let output_y = output_yb - batch_index * params.output_height;
    let output_x0 = tile_x * 32u + local_id.x;
    let output_x1 = output_x0 + 16u;
    let output_row0 = (batch_index * params.output_height + output_y)
        * params.output_width + output_x0;
    let output_row1 = output_row0 + 16u;
    let output_channel = local_id.y * 8u;
    let local_linear = local_id.y * 16u + local_id.x;
    let sparse_size = 81u * params.sparse_channel_count;
    var accum00 = vec4<f32>(0.0);
    var accum01 = vec4<f32>(0.0);
    var accum10 = vec4<f32>(0.0);
    var accum11 = vec4<f32>(0.0);

    for (var sparse_base = 0u; sparse_base < sparse_size; sparse_base += 32u) {
        let tile_k = local_linear & 31u;
        let sparse_k = sparse_base + tile_k;
        var input_channel = 0u;
        var kernel_spatial = 0u;
        if sparse_k < sparse_size {
            let channel_index = sparse_k % params.sparse_channel_count;
            kernel_spatial = sparse_k / params.sparse_channel_count;
            input_channel = bitcast<u32>(weights[params.sparse_channels_offset + channel_index]);
        }
        let kernel_y = kernel_spatial / 9u;
        let kernel_x = kernel_spatial - kernel_y * 9u;
        var tile_row = local_linear / 32u;
        loop {
            if tile_row >= 32u { break; }
            let output_x = tile_x * 32u + tile_row;
            var value = 0.0;
            if sparse_k < sparse_size && output_x < params.output_width {
                let input_y = i32(output_y + kernel_y) - 4;
                let input_x = i32(output_x + kernel_x) - 4;
                if input_y >= 0 && input_x >= 0
                    && input_y < i32(params.input_height)
                    && input_x < i32(params.input_width) {
                    value = arena[params.input_offset
                        + (((batch_index * params.input_height + u32(input_y)) * params.input_width
                        + u32(input_x)) * params.input_channel_stride)
                        + input_channel];
                }
            }
            input_tile[tile_row * 33u + tile_k] = value;
            tile_row += 4u;
        }

        var weight_index = local_linear;
        loop {
            if weight_index >= 512u { break; }
            let tile_weight_k = weight_index / 16u;
            let channel4 = weight_index - tile_weight_k * 16u;
            let item = sparse_base + tile_weight_k;
            var value = vec4<f32>(0.0);
            if item < sparse_size {
                let channel_index = item % params.sparse_channel_count;
                let spatial_index = item / params.sparse_channel_count;
                let source_channel = bitcast<u32>(
                    weights[params.sparse_channels_offset + channel_index]
                );
                let source = params.weight_offset
                    + (spatial_index * params.input_channels + source_channel)
                    * params.weight_k_stride
                    + channel4 * 4u;
                value = vec4<f32>(
                    weights[source],
                    weights[source + 1u],
                    weights[source + 2u],
                    weights[source + 3u],
                );
            }
            weight_tile[weight_index] = value;
            weight_index += 128u;
        }
        workgroupBarrier();

        let input_base0 = local_id.x * 33u;
        let input_base1 = (local_id.x + 16u) * 33u;
        let weight_lane = local_id.y * 2u;
        for (var k = 0u; k < 32u; k += 1u) {
            let weight_base = k * 16u + weight_lane;
            let weight0 = weight_tile[weight_base];
            let weight1 = weight_tile[weight_base + 1u];
            let input0 = input_tile[input_base0 + k];
            let input1 = input_tile[input_base1 + k];
            accum00 += input0 * weight0;
            accum01 += input0 * weight1;
            accum10 += input1 * weight0;
            accum11 += input1 * weight1;
        }
        workgroupBarrier();
    }

    if output_x0 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row0, output_channel + lane, accum00[lane]);
            write_output(output_row0, output_channel + lane + 4u, accum01[lane]);
        }
    }
    if output_x1 < params.output_width {
        for (var lane = 0u; lane < 4u; lane += 1u) {
            write_output(output_row1, output_channel + lane, accum10[lane]);
            write_output(output_row1, output_channel + lane + 4u, accum11[lane]);
        }
    }
}
