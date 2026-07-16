// Device-resident RGB preprocessing for PP-OCR. The source image is uploaded
// once as packed RGB8 pixels; this pass writes directly into the model's NHWC4
// activation arena so no host F32 tensor is created or uploaded.

struct ImagePreprocessParams {
    destination_offset: u32,
    source_width: u32,
    source_height: u32,
    destination_width: u32,
    destination_height: u32,
    content_width: u32,
    destination_channel_stride: u32,
    kind: u32,
    corner0_x_bits: u32,
    corner0_y_bits: u32,
    corner1_x_bits: u32,
    corner1_y_bits: u32,
    corner2_x_bits: u32,
    corner2_y_bits: u32,
    corner3_x_bits: u32,
    corner3_y_bits: u32,
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
}

@group(0) @binding(0)
var<storage, read> source: array<u32>;

@group(0) @binding(1)
var<storage, read_write> arena: array<f32>;

var<immediate> params: ImagePreprocessParams;

fn unpack_rgb(pixel: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(pixel & 255u),
        f32((pixel >> 8u) & 255u),
        f32((pixel >> 16u) & 255u),
    ) / 255.0;
}

fn source_pixel(x: u32, y: u32) -> vec3<f32> {
    return unpack_rgb(source[y * params.source_width + x]);
}

fn sample_bilinear(point: vec2<f32>) -> vec3<f32> {
    let max_x = f32(params.source_width - 1u);
    let max_y = f32(params.source_height - 1u);
    let clamped = clamp(point, vec2<f32>(0.0), vec2<f32>(max_x, max_y));
    let x0 = u32(floor(clamped.x));
    let y0 = u32(floor(clamped.y));
    let x1 = min(x0 + 1u, params.source_width - 1u);
    let y1 = min(y0 + 1u, params.source_height - 1u);
    let tx = clamped.x - f32(x0);
    let ty = clamped.y - f32(y0);
    let top = mix(source_pixel(x0, y0), source_pixel(x1, y0), tx);
    let bottom = mix(source_pixel(x0, y1), source_pixel(x1, y1), tx);
    return mix(top, bottom, ty);
}

fn normalize_bgr(bgr: vec3<f32>, channel: u32) -> f32 {
    if params.kind == 0u {
        switch channel {
            case 0u: { return (bgr.x - 0.485) / 0.229; }
            case 1u: { return (bgr.y - 0.456) / 0.224; }
            default: { return (bgr.z - 0.406) / 0.225; }
        }
    }
    return (bgr[channel] - 0.5) / 0.5;
}

@compute @workgroup_size(16, 16, 1)
fn preprocess(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if x >= params.destination_width || y >= params.destination_height {
        return;
    }
    let output = params.destination_offset
        + (y * params.destination_width + x) * params.destination_channel_stride;
    if x >= params.content_width {
        for (var channel = 0u; channel < params.destination_channel_stride; channel += 1u) {
            arena[output + channel] = 0.0;
        }
        return;
    }

    let c0 = vec2<f32>(
        bitcast<f32>(params.corner0_x_bits),
        bitcast<f32>(params.corner0_y_bits),
    );
    let c1 = vec2<f32>(
        bitcast<f32>(params.corner1_x_bits),
        bitcast<f32>(params.corner1_y_bits),
    );
    let c2 = vec2<f32>(
        bitcast<f32>(params.corner2_x_bits),
        bitcast<f32>(params.corner2_y_bits),
    );
    let c3 = vec2<f32>(
        bitcast<f32>(params.corner3_x_bits),
        bitcast<f32>(params.corner3_y_bits),
    );
    let u = (f32(x) + 0.5) / f32(params.content_width);
    let v = (f32(y) + 0.5) / f32(params.destination_height);
    let top = mix(c0, c1, u);
    let bottom = mix(c3, c2, u);
    let rgb = sample_bilinear(mix(top, bottom, v) - vec2<f32>(0.5));
    let bgr = vec3<f32>(rgb.z, rgb.y, rgb.x);
    arena[output] = normalize_bgr(bgr, 0u);
    arena[output + 1u] = normalize_bgr(bgr, 1u);
    arena[output + 2u] = normalize_bgr(bgr, 2u);
    for (var channel = 3u; channel < params.destination_channel_stride; channel += 1u) {
        arena[output + channel] = 0.0;
    }
}
