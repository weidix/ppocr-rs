// GPU runtime: direct NHWC stride-2 3x3 convolution for the medium detector stem.
// Each workgroup computes an 8x8 spatial tile and 32 output channels.

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

// 17x17 input patch with four channels, plus 3x3x4x32 weights.
var<workgroup> input_tile: array<f32, 1156>;
var<workgroup> weight_tile: array<vec4<f32>, 288>;

fn write_output(output_row: u32, channel: u32, value: f32) {
    var result = value;
    if (params.flags & 1u) != 0u {
        result += weights[params.bias_offset + channel];
    }
    arena[params.output_offset + output_row * params.output_channel_stride + channel] =
        max(result, 0.0);
}

@compute @workgroup_size(8, 8, 2)
fn conv_3x3_stride2_direct(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tile_group = workgroup_id.z * params.reserved0 + workgroup_id.x;
    let tiles_per_row = params.reserved1;
    let tile_rows = (params.output_height + 7u) / 8u;
    let tile_count = params.batch * tile_rows * tiles_per_row;
    if tile_group >= tile_count || workgroup_id.y >= 2u {
        return;
    }

    let tile_x = tile_group % tiles_per_row;
    let tile_yb = tile_group / tiles_per_row;
    let batch_index = tile_yb / tile_rows;
    let tile_y = tile_yb - batch_index * tile_rows;
    let output_x = tile_x * 8u + local_id.x;
    let output_y = tile_y * 8u + local_id.y;
    let output_channel = workgroup_id.y * 32u + local_id.z * 16u;
    let local_linear = (local_id.z * 8u + local_id.y) * 8u + local_id.x;
    var accum0 = vec4<f32>(0.0);
    var accum1 = vec4<f32>(0.0);
    var accum2 = vec4<f32>(0.0);
    var accum3 = vec4<f32>(0.0);

    for (var input_channel_base = 0u;
        input_channel_base < params.input_channels;
        input_channel_base += 4u) {
        var input_index = local_linear;
        loop {
            if input_index >= 1156u { break; }
            let patch_point = input_index / 4u;
            let patch_channel = input_index - patch_point * 4u;
            let patch_y = patch_point / 17u;
            let patch_x = patch_point - patch_y * 17u;
            let input_y = i32(tile_y * 16u + patch_y) - 1;
            let input_x = i32(tile_x * 16u + patch_x) - 1;
            var value = 0.0;
            if input_y >= 0 && input_x >= 0
                && input_y < i32(params.input_height)
                && input_x < i32(params.input_width) {
                value = arena[params.input_offset
                    + (((batch_index * params.input_height + u32(input_y)) * params.input_width
                    + u32(input_x)) * params.input_channel_stride)
                    + input_channel_base + patch_channel];
            }
            input_tile[input_index] = value;
            input_index += 128u;
        }

        var weight_index = local_linear;
        loop {
            if weight_index >= 288u { break; }
            let local_k = weight_index / 8u;
            let channel4 = weight_index - local_k * 8u;
            let kernel_spatial = local_k / 4u;
            let input_channel = input_channel_base + local_k - kernel_spatial * 4u;
            let source = params.weight_offset
                + (kernel_spatial * params.input_channels + input_channel)
                    * params.weight_k_stride
                + workgroup_id.y * 32u
                + channel4 * 4u;
            weight_tile[weight_index] = vec4<f32>(
                weights[source],
                weights[source + 1u],
                weights[source + 2u],
                weights[source + 3u],
            );
            weight_index += 128u;
        }

        workgroupBarrier();

        let channel4_base = local_id.z * 4u;
        for (var kernel_y = 0u; kernel_y < 3u; kernel_y += 1u) {
            for (var kernel_x = 0u; kernel_x < 3u; kernel_x += 1u) {
                let input_base = (((local_id.y * 2u + kernel_y) * 17u
                    + local_id.x * 2u + kernel_x) * 4u);
                let weight_base = (kernel_y * 3u + kernel_x) * 32u + channel4_base;
                for (var input_channel = 0u; input_channel < 4u; input_channel += 1u) {
                    let value = input_tile[input_base + input_channel];
                    let base = weight_base + input_channel * 8u;
                    accum0 += value * weight_tile[base];
                    accum1 += value * weight_tile[base + 1u];
                    accum2 += value * weight_tile[base + 2u];
                    accum3 += value * weight_tile[base + 3u];
                }
            }
        }

        workgroupBarrier();
    }

    if output_y >= params.output_height || output_x >= params.output_width {
        return;
    }
    let output_row = (batch_index * params.output_height + output_y)
        * params.output_width + output_x;
    for (var lane = 0u; lane < 4u; lane += 1u) {
        write_output(output_row, output_channel + lane, accum0[lane]);
        write_output(output_row, output_channel + lane + 4u, accum1[lane]);
        write_output(output_row, output_channel + lane + 8u, accum2[lane]);
        write_output(output_row, output_channel + lane + 12u, accum3[lane]);
    }
}
