//! Architecture-specific CPU kernels.

use rayon::prelude::*;

#[cfg(target_os = "macos")]
mod accelerate;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[inline]
pub(crate) fn supports_exact_sparse_gemm() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2_fma()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

#[inline]
pub(crate) fn supports_stride2_simd_copy() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2_fma()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub(crate) fn supports_direct_spatial_conv() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2_fma()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
pub(crate) unsafe fn copy_stride2_16(output: *mut f32, input: *const f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        x86::copy_stride2_16(output, input);
    }
    #[cfg(not(target_arch = "x86_64"))]
    for lane in 0..16 {
        unsafe { *output.add(lane) = *input.add(lane * 2) };
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn spatial_conv2d_direct(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    input_channels: usize,
    input_height: usize,
    input_width: usize,
    output_channels: usize,
    output_height: usize,
    output_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    activation: Option<UnaryOperation>,
) {
    const BLOCK_ROWS: usize = 4;
    assert!(supports_direct_spatial_conv());
    assert!(output_channels.is_multiple_of(BLOCK_ROWS));
    assert_eq!(input.len(), input_channels * input_height * input_width);
    assert_eq!(output.len(), output_channels * output_height * output_width);
    let patch_size = input_channels * kernel_height * kernel_width;
    assert_eq!(weight.len(), output_channels * patch_size);
    assert!(bias.is_none_or(|bias| bias.len() == output_channels));
    let output_plane = output_height * output_width;

    output
        .par_chunks_mut(BLOCK_ROWS * output_plane)
        .enumerate()
        .for_each(|(block, output)| {
            let row_start = block * BLOCK_ROWS;
            let weight_start = block * BLOCK_ROWS * patch_size;
            let weight = &weight[weight_start..weight_start + BLOCK_ROWS * patch_size];
            let bias = bias.map(|bias| &bias[row_start..row_start + BLOCK_ROWS]);
            #[cfg(target_arch = "x86_64")]
            // SAFETY: Runtime AVX2/FMA support and all matrix/image dimensions
            // are validated above. The kernel handles borders without OOB loads.
            unsafe {
                x86::spatial_conv2d_packed_4(
                    output,
                    input,
                    weight,
                    input_channels,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                    kernel_height,
                    kernel_width,
                    strides,
                    pads,
                    bias,
                    activation,
                )
            };
        });
}

#[inline]
pub(crate) fn fill(values: &mut [f32], value: f32) {
    values.fill(value);
}

#[inline]
pub(crate) fn axpy(output: &mut [f32], input: &[f32], scale: f32) {
    assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The implementation only performs unaligned loads/stores within
        // the bounds of equally sized slices. NEON is mandatory on AArch64.
        unsafe { neon::axpy(output, input, scale) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime, and equal-length
        // slices bound every unaligned vector load and store.
        unsafe { x86::axpy(output, input, scale) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (output, input) in output.iter_mut().zip(input) {
        *output = input.mul_add(scale, *output);
    }
}

#[inline]
pub(crate) fn add_in_place(output: &mut [f32], input: &[f32]) {
    axpy(output, input, 1.0);
}

pub(crate) fn depthwise_conv2d_same(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    kernel: usize,
    bias: f32,
) {
    assert!(height > 0 && width > 0);
    assert!(matches!(kernel, 3 | 5 | 7 | 9));
    assert_eq!(output.len(), height * width);
    assert_eq!(input.len(), height * width);
    assert_eq!(weights.len(), kernel * kernel);

    macro_rules! dispatch {
        ($kernel:literal) => {{
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: Slice dimensions are checked above. The NEON kernel
                // vectorizes only columns whose complete KxK window is in bounds.
                unsafe {
                    neon::depthwise_conv2d_same::<$kernel>(
                        output, input, weights, height, width, bias,
                    )
                };
            }
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma() {
                // SAFETY: AVX2 and FMA were detected at runtime. Slice dimensions
                // are checked above, and the kernel vectorizes only complete
                // interior windows.
                unsafe {
                    x86::depthwise_conv2d_same::<$kernel>(
                        output, input, weights, height, width, bias,
                    )
                };
                return;
            }
            #[cfg(not(target_arch = "aarch64"))]
            depthwise_conv2d_same_scalar::<$kernel>(output, input, weights, height, width, bias);
        }};
    }

    match kernel {
        3 => dispatch!(3),
        5 => dispatch!(5),
        7 => dispatch!(7),
        9 => dispatch!(9),
        _ => unreachable!(),
    }
}

pub(crate) fn depthwise_conv2d_same_3x3_stride2(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    bias: f32,
) {
    assert!(height > 0 && width > 0);
    assert_eq!(output.len(), height.div_ceil(2) * width.div_ceil(2));
    assert_eq!(input.len(), height * width);
    assert_eq!(weights.len(), 9);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: Slice dimensions are checked above. The NEON kernel only
    // vectorizes complete interior windows and handles borders separately.
    unsafe {
        neon::depthwise_conv2d_same_3x3_stride2(output, input, weights, height, width, bias)
    };
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime. Slice dimensions are
        // checked above, and complete interior vectors are bounded by the input.
        unsafe {
            x86::depthwise_conv2d_same_3x3_stride2(output, input, weights, height, width, bias)
        };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let output_width = width.div_ceil(2);
        for output_y in 0..height.div_ceil(2) {
            for output_x in 0..output_width {
                let mut sum = bias;
                for kernel_y in 0..3 {
                    let input_y = output_y * 2 + kernel_y;
                    if input_y == 0 || input_y > height {
                        continue;
                    }
                    for kernel_x in 0..3 {
                        let input_x = output_x * 2 + kernel_x;
                        if input_x == 0 || input_x > width {
                            continue;
                        }
                        sum = input[(input_y - 1) * width + input_x - 1]
                            .mul_add(weights[kernel_y * 3 + kernel_x], sum);
                    }
                }
                output[output_y * output_width + output_x] = sum;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn depthwise_conv2d_same_scalar<const K: usize>(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    height: usize,
    width: usize,
    bias: f32,
) {
    let padding = K / 2;
    for y in 0..height {
        let kernel_y_start = padding.saturating_sub(y);
        let kernel_y_end = K.min(height + padding - y);
        for x in 0..width {
            let kernel_x_start = padding.saturating_sub(x);
            let kernel_x_end = K.min(width + padding - x);
            let mut sum = bias;
            for kernel_y in kernel_y_start..kernel_y_end {
                let input_y = y + kernel_y - padding;
                for kernel_x in kernel_x_start..kernel_x_end {
                    let input_x = x + kernel_x - padding;
                    sum = input[input_y * width + input_x]
                        .mul_add(weights[kernel_y * K + kernel_x], sum);
                }
            }
            output[y * width + x] = sum;
        }
    }
}

pub(crate) fn mul_in_place(output: &mut [f32], input: &[f32]) {
    assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Equal-length slices bound every vector load and store.
        unsafe { neon::mul_in_place(output, input) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime, and both slices have
        // the same length.
        unsafe { x86::mul_in_place(output, input) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (output, input) in output.iter_mut().zip(input) {
        *output *= *input;
    }
}

pub(crate) fn affine_in_place(values: &mut [f32], scale: f32, bias: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The in-place kernel only accesses the supplied slice.
        unsafe { neon::affine(values, scale, bias) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel stays
        // within the supplied slice.
        unsafe { x86::affine(values, scale, bias) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        *value = value.mul_add(scale, bias);
    }
}

pub(crate) fn residual_mul_in_place(values: &mut [f32], gate: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The in-place kernel only accesses the supplied slice.
        unsafe { neon::residual_mul(values, gate) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel stays
        // within the supplied slice.
        unsafe { x86::residual_mul(values, gate) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        let original = *value;
        let scaled = original.mul_add(gate, 0.0);
        *value = scaled.mul_add(1.0, original);
    }
}

pub(crate) fn square_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The in-place kernel only accesses the supplied slice.
        unsafe { neon::square(values) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel stays
        // within the supplied slice.
        unsafe { x86::square(values) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        *value *= *value;
    }
}

pub(crate) fn gemm(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
) {
    gemm_with_activation(output, left, right, rows, inner, columns, bias, None);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_with_activation(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    gemm_impl(
        output, left, right, rows, inner, columns, bias, None, false, activation, false,
    );
}

#[cfg(test)]
pub(crate) fn gemm_packed_left(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
) {
    gemm_packed_left_with_activation(output, left, right, rows, inner, columns, bias, None);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_packed_left_with_activation(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    gemm_impl(
        output, left, right, rows, inner, columns, bias, None, true, activation, false,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_system_dense(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert!(bias.is_none_or(|bias| bias.len() == rows));
    #[cfg(target_os = "macos")]
    {
        accelerate::sgemm(output, left, right, rows, inner, columns);
        output
            .par_chunks_mut(columns)
            .enumerate()
            .for_each(|(row, output)| {
                if let Some(bias) = bias {
                    affine_in_place(output, 1.0, bias[row]);
                }
                if let Some(activation) = activation {
                    unary_chunk(output, activation);
                }
            });
    }
    #[cfg(not(target_os = "macos"))]
    gemm_with_activation(output, left, right, rows, inner, columns, bias, activation);
}

#[allow(clippy::too_many_arguments)]
#[cfg(target_os = "macos")]
pub(crate) fn linear_system_dense(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    softmax: bool,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert!(bias.is_none_or(|bias| bias.len() == columns));
    accelerate::sgemm_right_transposed(output, input, weight, rows, inner, columns);
    output.par_chunks_mut(columns).for_each(|row| {
        if softmax {
            if let Some(bias) = bias {
                bias_softmax_in_place(row, bias);
            } else {
                softmax_in_place(row);
            }
        } else if let Some(bias) = bias {
            add_in_place(row, bias);
        }
    });
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_os = "macos"))]
pub(crate) fn linear_right_transposed(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
    softmax: bool,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert_eq!(output.len(), rows * columns);
    assert_eq!(input.len(), rows * inner);
    assert_eq!(weight.len(), columns * inner);
    assert!(bias.is_none_or(|bias| bias.len() == columns));
    assert!(!softmax || activation.is_none());

    const MICRO_ROWS: usize = 8;
    let row_blocks = rows.div_ceil(MICRO_ROWS);
    let blocks_per_task = row_blocks.div_ceil(rayon::current_num_threads()).max(1);
    let task_rows = blocks_per_task * MICRO_ROWS;
    output
        .par_chunks_mut(task_rows * columns)
        .enumerate()
        .for_each(|(task, output)| {
            let row_start = task * task_rows;
            let task_row_count = (rows - row_start).min(task_rows);
            for local_row in (0..task_row_count).step_by(MICRO_ROWS) {
                let block_rows = (task_row_count - local_row).min(MICRO_ROWS);
                let input_start = (row_start + local_row) * inner;
                let input = &input[input_start..input_start + block_rows * inner];
                let output = &mut output[local_row * columns..(local_row + block_rows) * columns];

                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma() {
                    macro_rules! dispatch_rows {
                        ($rows:literal) => {
                            // SAFETY: AVX2/FMA were checked above. All slices
                            // contain the complete row block described here.
                            unsafe {
                                x86::linear_rows_8::<$rows>(
                                    output, input, weight, inner, columns, bias, activation,
                                )
                            }
                        };
                    }
                    match block_rows {
                        1 => dispatch_rows!(1),
                        2 => dispatch_rows!(2),
                        3 => dispatch_rows!(3),
                        4 => dispatch_rows!(4),
                        5 => dispatch_rows!(5),
                        6 => dispatch_rows!(6),
                        7 => dispatch_rows!(7),
                        8 => dispatch_rows!(8),
                        _ => unreachable!(),
                    }
                } else {
                    linear_rows_scalar(
                        output, input, weight, block_rows, inner, columns, bias, activation,
                    );
                }

                #[cfg(not(target_arch = "x86_64"))]
                linear_rows_scalar(
                    output, input, weight, block_rows, inner, columns, bias, activation,
                );

                if softmax {
                    output.chunks_mut(columns).for_each(softmax_in_place);
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_os = "macos"))]
fn linear_rows_scalar(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    #[cfg(target_arch = "x86_64")]
    for column_start in (0..columns).step_by(8) {
        let block_columns = (columns - column_start).min(8);
        let weight = &weight[column_start * inner..(column_start + block_columns) * inner];
        for row in 0..rows {
            for column in 0..block_columns {
                let mut sum = bias.map_or(0.0, |bias| bias[column_start + column]);
                for index in 0..inner {
                    sum = input[row * inner + index]
                        .mul_add(weight[index * block_columns + column], sum);
                }
                output[row * columns + column_start + column] =
                    activation.map_or(sum, |activation| activation.apply(sum));
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = bias.map_or(0.0, |bias| bias[column]);
            for index in 0..inner {
                sum = input[row * inner + index].mul_add(weight[column * inner + index], sum);
            }
            output[row * columns + column] =
                activation.map_or(sum, |activation| activation.apply(sum));
        }
    }
}

pub(crate) fn gemm_column_bias_softmax(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    column_bias: &[f32],
) {
    gemm_impl(
        output,
        left,
        right,
        rows,
        inner,
        columns,
        None,
        Some(column_bias),
        false,
        None,
        true,
    );
}

#[allow(clippy::too_many_arguments)]
fn gemm_impl(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
    packed_left: bool,
    activation: Option<UnaryOperation>,
    row_softmax: bool,
) {
    assert!(rows > 0 && inner > 0 && columns > 0);
    assert_eq!(rows.checked_mul(columns), Some(output.len()));
    assert_eq!(rows.checked_mul(inner), Some(left.len()));
    assert_eq!(inner.checked_mul(columns), Some(right.len()));
    assert!(bias.is_none_or(|bias| bias.len() == rows));
    assert!(column_bias.is_none_or(|bias| bias.len() == columns));
    assert!(bias.is_none() || column_bias.is_none());
    assert!(!packed_left || column_bias.is_none());
    let micro_rows = if packed_left { 12 } else { 8 };
    let row_blocks = rows.div_ceil(micro_rows);
    let blocks_per_task = row_blocks.div_ceil(rayon::current_num_threads()).max(1);
    let task_rows = blocks_per_task * micro_rows;
    output
        .par_chunks_mut(task_rows * columns)
        .enumerate()
        .for_each(|(task, output)| {
            let task_row_start = task * task_rows;
            let task_row_count = (rows - task_row_start).min(task_rows);
            for local_row in (0..task_row_count).step_by(micro_rows) {
                let block_rows = (task_row_count - local_row).min(micro_rows);
                let row_start = task_row_start + local_row;
                let output = &mut output[local_row * columns..(local_row + block_rows) * columns];
                let left = &left[row_start * inner..(row_start + block_rows) * inner];
                let bias = bias.map(|bias| &bias[row_start..row_start + block_rows]);
                gemm_rows(
                    output,
                    left,
                    right,
                    block_rows,
                    inner,
                    columns,
                    columns,
                    bias,
                    column_bias,
                    packed_left,
                    activation,
                );
                if row_softmax {
                    for row in output.chunks_mut(columns) {
                        softmax_in_place(row);
                    }
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_packed_panels(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    panels: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    const PANEL_COLUMNS: usize = 16;

    assert_eq!(output.len(), panels * rows * PANEL_COLUMNS);
    assert_eq!(left.len(), rows * inner);
    assert_eq!(right.len(), panels * inner * PANEL_COLUMNS);
    #[cfg(target_arch = "aarch64")]
    if rows.is_multiple_of(4) {
        gemm_packed_panels_blocked(output, left, right, rows, inner, panels, bias, activation);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if rows.is_multiple_of(4) && has_avx2_fma() {
        gemm_packed_panels_blocked(output, left, right, rows, inner, panels, bias, activation);
        return;
    }
    output
        .par_chunks_mut(rows * PANEL_COLUMNS)
        .zip(right.par_chunks(inner * PANEL_COLUMNS))
        .for_each(|(output, right)| {
            for row_start in (0..rows).step_by(4) {
                let block_rows = (rows - row_start).min(4);
                let output = &mut output
                    [row_start * PANEL_COLUMNS..(row_start + block_rows) * PANEL_COLUMNS];
                let left = &left[row_start * inner..(row_start + block_rows) * inner];
                let bias = bias.map(|bias| &bias[row_start..row_start + block_rows]);
                gemm_rows(
                    output,
                    left,
                    right,
                    block_rows,
                    inner,
                    PANEL_COLUMNS,
                    PANEL_COLUMNS,
                    bias,
                    None,
                    true,
                    activation,
                );
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_sparse_packed_panels(
    output: &mut [f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    panels: usize,
    bias: Option<&[f32]>,
    row_offsets: &[usize],
    indices: &[u32],
    values: &[f32],
    activation: Option<UnaryOperation>,
) {
    const PANEL_COLUMNS: usize = 16;
    const BLOCK_ROWS: usize = 4;

    assert_eq!(output.len(), panels * rows * PANEL_COLUMNS);
    assert_eq!(right.len(), panels * inner * PANEL_COLUMNS);
    assert!(rows.is_multiple_of(BLOCK_ROWS));
    assert_eq!(row_offsets.len(), rows / BLOCK_ROWS + 1);
    assert_eq!(values.len(), indices.len() * BLOCK_ROWS);
    output
        .par_chunks_mut(rows * PANEL_COLUMNS)
        .zip(right.par_chunks(inner * PANEL_COLUMNS))
        .for_each(|(output, right)| {
            for block in 0..rows / BLOCK_ROWS {
                let row_start = block * BLOCK_ROWS;
                let entry_start = row_offsets[block];
                let entry_end = row_offsets[block + 1];
                let output = &mut output
                    [row_start * PANEL_COLUMNS..(row_start + BLOCK_ROWS) * PANEL_COLUMNS];
                let bias = bias.map(|bias| &bias[row_start..row_start + BLOCK_ROWS]);
                #[cfg(target_arch = "x86_64")]
                if has_avx2_fma() {
                    // SAFETY: AVX2/FMA were checked above. Every sparse index
                    // selects a complete packed RHS row.
                    unsafe {
                        x86::gemm_4x16_sparse(
                            output,
                            right,
                            &indices[entry_start..entry_end],
                            &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                            PANEL_COLUMNS,
                            PANEL_COLUMNS,
                            bias,
                            activation,
                        )
                    };
                    continue;
                }
                #[cfg(target_arch = "aarch64")]
                // SAFETY: Indices reference complete 16-column rows in the
                // packed RHS, and weights contain four values per entry.
                unsafe {
                    neon::gemm_4x16_sparse(
                        output,
                        right,
                        &indices[entry_start..entry_end],
                        &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                        PANEL_COLUMNS,
                        PANEL_COLUMNS,
                        bias,
                    );
                    if let Some(activation) = activation {
                        unary_chunk(output, activation);
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                gemm_4_sparse_scalar(
                    output,
                    right,
                    &indices[entry_start..entry_end],
                    &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                    PANEL_COLUMNS,
                    PANEL_COLUMNS,
                    bias,
                    activation,
                );
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_sparse_packed_left(
    output: &mut [f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
    row_offsets: &[usize],
    indices: &[u32],
    values: &[f32],
    activation: Option<UnaryOperation>,
) {
    const BLOCK_ROWS: usize = 4;

    assert_eq!(output.len(), rows * columns);
    assert_eq!(right.len(), inner * columns);
    assert!(rows.is_multiple_of(BLOCK_ROWS));
    assert_eq!(row_offsets.len(), rows / BLOCK_ROWS + 1);
    assert_eq!(values.len(), indices.len() * BLOCK_ROWS);
    output
        .par_chunks_mut(BLOCK_ROWS * columns)
        .enumerate()
        .for_each(|(block, output)| {
            let entry_start = row_offsets[block];
            let entry_end = row_offsets[block + 1];
            let row_start = block * BLOCK_ROWS;
            let bias = bias.map(|bias| &bias[row_start..row_start + BLOCK_ROWS]);
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma() {
                // SAFETY: AVX2/FMA were checked above. Every sparse index
                // selects a complete RHS row of `columns` values.
                unsafe {
                    x86::gemm_4x16_sparse(
                        output,
                        right,
                        &indices[entry_start..entry_end],
                        &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                        columns,
                        columns,
                        bias,
                        activation,
                    )
                };
                return;
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: Every sparse index identifies a complete RHS row and
            // weights contain four values per entry.
            unsafe {
                neon::gemm_4x16_sparse(
                    output,
                    right,
                    &indices[entry_start..entry_end],
                    &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                    columns,
                    columns,
                    bias,
                );
                if let Some(activation) = activation {
                    unary_chunk(output, activation);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            gemm_4_sparse_scalar(
                output,
                right,
                &indices[entry_start..entry_end],
                &values[entry_start * BLOCK_ROWS..entry_end * BLOCK_ROWS],
                columns,
                columns,
                bias,
                activation,
            );
        });
}

#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
fn gemm_4_sparse_scalar(
    output: &mut [f32],
    right: &[f32],
    indices: &[u32],
    values: &[f32],
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    for row in 0..4 {
        for column in 0..columns {
            let mut sum = bias.map_or(0.0, |bias| bias[row]);
            for (entry, &index) in indices.iter().enumerate() {
                sum = values[entry * 4 + row]
                    .mul_add(right[index as usize * right_stride + column], sum);
            }
            output[row * columns + column] =
                activation.map_or(sum, |activation| activation.apply(sum));
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn gemm_packed_panels_blocked(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    _panels: usize,
    bias: Option<&[f32]>,
    activation: Option<UnaryOperation>,
) {
    const PANEL_COLUMNS: usize = 16;
    const DEPTH_BLOCK: usize = 256;

    for depth_start in (0..inner).step_by(DEPTH_BLOCK) {
        let depth = (inner - depth_start).min(DEPTH_BLOCK);
        output
            .par_chunks_mut(rows * PANEL_COLUMNS)
            .zip(right.par_chunks(inner * PANEL_COLUMNS))
            .for_each(|(output, right)| {
                let right_start = depth_start * PANEL_COLUMNS;
                let right = &right[right_start..right_start + depth * PANEL_COLUMNS];
                for row_start in (0..rows).step_by(4) {
                    let output =
                        &mut output[row_start * PANEL_COLUMNS..(row_start + 4) * PANEL_COLUMNS];
                    let left_start = row_start * inner + depth_start * 4;
                    let left = &left[left_start..left_start + depth * 4];
                    let bias = bias.map(|bias| &bias[row_start..row_start + 4]);
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: AVX2/FMA availability is checked by the caller;
                    // slices describe one packed 4x16 tile.
                    unsafe {
                        x86::gemm_4x16_packed(
                            output,
                            left,
                            right,
                            depth,
                            PANEL_COLUMNS,
                            PANEL_COLUMNS,
                            bias,
                            depth_start != 0,
                            (depth_start + depth == inner)
                                .then_some(activation)
                                .flatten(),
                        )
                    };
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: The slices describe a complete 4x16 tile and
                    // NEON is mandatory on AArch64.
                    unsafe {
                        neon::gemm_4x16_packed(
                            output,
                            left,
                            right,
                            depth,
                            PANEL_COLUMNS,
                            PANEL_COLUMNS,
                            bias,
                            depth_start != 0,
                        )
                    };
                }
            });
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(activation) = activation {
        output
            .par_chunks_mut(rows * PANEL_COLUMNS)
            .for_each(|output| unary_chunk(output, activation));
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_rows(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
    packed_left: bool,
    activation: Option<UnaryOperation>,
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        if rows == 12 && packed_left {
            debug_assert!(column_bias.is_none());
            // SAFETY: AVX2 and FMA were detected at runtime. The caller
            // supplies twelve packed left rows and complete output rows.
            unsafe {
                x86::gemm_rows_8::<12, true>(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    None,
                    activation,
                )
            };
            return;
        }
        if rows == 8 {
            // SAFETY: AVX2 and FMA were detected at runtime. Slice dimensions
            // describe eight complete rows in the selected left layout.
            unsafe {
                if packed_left {
                    debug_assert!(column_bias.is_none());
                    x86::gemm_rows_8::<8, true>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        None,
                        activation,
                    )
                } else {
                    x86::gemm_rows_8::<8, false>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        column_bias,
                        activation,
                    )
                }
            };
            return;
        }
        if rows == 4 {
            // SAFETY: Same runtime feature and matrix-bounds argument as the
            // eight-row kernel above.
            unsafe {
                if packed_left {
                    debug_assert!(column_bias.is_none());
                    x86::gemm_rows_8::<4, true>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        None,
                        activation,
                    )
                } else {
                    x86::gemm_rows_8::<4, false>(
                        output,
                        left,
                        right,
                        inner,
                        columns,
                        right_stride,
                        bias,
                        column_bias,
                        activation,
                    )
                }
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if rows == 12 && packed_left {
        // SAFETY: The caller supplies exactly twelve complete output rows and
        // twelve interleaved weights for every inner-dimension position.
        debug_assert!(column_bias.is_none());
        const DEPTH_BLOCK: usize = 256;
        for depth_start in (0..inner).step_by(DEPTH_BLOCK) {
            let depth = (inner - depth_start).min(DEPTH_BLOCK);
            let packed_left = &left[depth_start * 12..(depth_start + depth) * 12];
            let right = &right[depth_start * right_stride..];
            unsafe {
                neon::gemm_12x8_packed(
                    output,
                    packed_left,
                    right,
                    depth,
                    columns,
                    right_stride,
                    bias,
                    depth_start != 0,
                )
            };
        }
        if let Some(activation) = activation {
            unary_chunk(output, activation);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if rows == 8 {
        // SAFETY: The caller supplies exactly eight complete output/left rows;
        // right_stride is validated by the owning full matrix in gemm.
        unsafe {
            if packed_left {
                debug_assert!(column_bias.is_none());
                neon::gemm_8x12_packed(output, left, right, inner, columns, right_stride, bias)
            } else {
                neon::gemm_8x12(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    column_bias,
                )
            }
        };
        if let Some(activation) = activation {
            unary_chunk(output, activation);
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if rows == 4 {
        // SAFETY: Same bounds argument as the eight-row kernel.
        unsafe {
            if packed_left {
                debug_assert!(column_bias.is_none());
                neon::gemm_4x16_packed(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    false,
                )
            } else {
                neon::gemm_4x16(
                    output,
                    left,
                    right,
                    inner,
                    columns,
                    right_stride,
                    bias,
                    column_bias,
                )
            }
        };
        if let Some(activation) = activation {
            unary_chunk(output, activation);
        }
        return;
    }
    gemm_scalar_strided(
        output,
        left,
        right,
        rows,
        inner,
        columns,
        right_stride,
        bias,
        column_bias,
        packed_left,
    );
    if let Some(activation) = activation {
        unary_chunk(output, activation);
    }
}

pub(crate) fn max_pool_2x2_same_upper(output: &mut [f32], input: &[f32], width: usize) {
    assert_eq!(output.len(), input.len());
    assert!(width > 0 && input.len().is_multiple_of(width));
    let height = input.len() / width;
    for y in 0..height {
        let current = &input[y * width..(y + 1) * width];
        let next = (y + 1 < height).then(|| &input[(y + 1) * width..(y + 2) * width]);
        let output = &mut output[y * width..(y + 1) * width];
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: Rows have identical lengths and the kernel handles the
            // right edge without reading past either row.
            unsafe { neon::max_pool_2x2_row(output, current, next) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        for x in 0..width {
            let mut maximum = current[x];
            if x + 1 < width {
                maximum = maximum.max(current[x + 1]);
            }
            if let Some(next) = next {
                maximum = maximum.max(next[x]);
                if x + 1 < width {
                    maximum = maximum.max(next[x + 1]);
                }
            }
            output[x] = maximum;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_scalar_strided(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
    packed_left: bool,
) {
    for row in 0..rows {
        let output = &mut output[row * columns..(row + 1) * columns];
        if let Some(column_bias) = column_bias {
            output.copy_from_slice(column_bias);
        } else {
            output.fill(bias.map_or(0.0, |bias| bias[row]));
        }
        for index in 0..inner {
            axpy(
                output,
                &right[index * right_stride..index * right_stride + columns],
                if packed_left {
                    left[index * rows + row]
                } else {
                    left[row * inner + index]
                },
            );
        }
    }
}

pub(crate) fn unary_in_place(values: &mut [f32], operation: UnaryOperation) {
    const PARALLEL_CHUNK: usize = 32 * 1024;
    if values.len() <= PARALLEL_CHUNK {
        unary_chunk(values, operation);
        return;
    }
    values.par_chunks_mut(PARALLEL_CHUNK).for_each(|values| {
        unary_chunk(values, operation);
    });
}

pub(crate) fn softmax_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The kernel reads and writes only complete vectors inside the
        // supplied slice and handles the remaining values with safe indexing.
        unsafe { neon::softmax(values) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the kernel only
        // accesses full vectors and a scalar tail inside the slice.
        unsafe { x86::softmax(values) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for value in values.iter_mut() {
            *value = (*value - maximum).exp();
            sum += *value;
        }
        let reciprocal = sum.recip();
        for value in values {
            *value *= reciprocal;
        }
    }
}

pub(crate) fn bias_softmax_in_place(values: &mut [f32], bias: &[f32]) {
    assert_eq!(values.len(), bias.len());
    #[cfg(target_arch = "aarch64")]
    // SAFETY: Equal-length slices bound all vector loads and stores.
    unsafe {
        neon::bias_softmax(values, bias)
    };
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2/FMA were detected and equal-length slices bound every
        // load and store, including the scalar tail.
        unsafe { x86::bias_softmax(values, bias) };
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        add_in_place(values, bias);
        softmax_in_place(values);
    }
}

pub(crate) fn mean(values: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    let sum = {
        // SAFETY: The kernel only loads complete vectors within the slice and
        // handles its tail through safe indexing.
        unsafe { neon::sum(values) }
    };
    #[cfg(target_arch = "x86_64")]
    let sum = if has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at runtime; the reduction only
        // reads full vectors and a scalar tail inside the slice.
        unsafe { x86::sum(values) }
    } else {
        values.iter().copied().sum::<f32>()
    };
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let sum = values.iter().copied().sum::<f32>();
    sum / values.len() as f32
}

pub(crate) fn layer_norm_in_place(values: &mut [f32], weight: &[f32], bias: &[f32], epsilon: f32) {
    assert!(!values.is_empty());
    assert_eq!(values.len(), weight.len());
    assert_eq!(values.len(), bias.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: Runtime feature detection covers the AVX2/FMA kernel and all
        // three slices have the same validated length.
        unsafe { x86::layer_norm(values, weight, bias, epsilon) };
        return;
    }

    let mean = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let inverse_std = (variance + epsilon).sqrt().recip();
    for ((value, weight), bias) in values.iter_mut().zip(weight).zip(bias) {
        *value = (*value - mean).mul_add(inverse_std * *weight, *bias);
    }
}

fn unary_chunk(values: &mut [f32], operation: UnaryOperation) {
    #[cfg(target_arch = "aarch64")]
    {
        match operation {
            UnaryOperation::Relu => {
                // SAFETY: The operation is in-place and stays in slice bounds.
                unsafe { neon::relu(values) };
                return;
            }
            UnaryOperation::Gelu => {
                // SAFETY: The operation is in-place and stays in slice bounds.
                unsafe { neon::gelu(values) };
                return;
            }
            _ => {}
        }
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        match operation {
            UnaryOperation::Relu => {
                // SAFETY: AVX2 and FMA were detected at runtime, and the
                // operation stays within the supplied slice.
                unsafe { x86::relu(values) };
                return;
            }
            UnaryOperation::Gelu => {
                // SAFETY: Same feature and slice-bounds argument as ReLU.
                unsafe { x86::gelu(values) };
                return;
            }
            UnaryOperation::Silu => {
                // SAFETY: Same feature and slice-bounds argument as ReLU.
                unsafe { x86::silu(values) };
                return;
            }
            _ => {}
        }
    }
    for value in values {
        *value = operation.apply(*value);
    }
}

#[derive(Clone, Copy)]
pub(crate) enum UnaryOperation {
    Relu,
    Erf,
    Gelu,
    HardSwish,
    Sigmoid,
    Silu,
    Sqrt,
    HardSigmoid { alpha: f32, beta: f32 },
}

impl UnaryOperation {
    #[inline]
    pub(super) fn apply(self, value: f32) -> f32 {
        match self {
            Self::Relu => value.max(0.0),
            Self::Erf => erf(value),
            Self::Gelu => 0.5 * value * (1.0 + erf(value * std::f32::consts::FRAC_1_SQRT_2)),
            Self::HardSwish => value * (value / 6.0 + 0.5).clamp(0.0, 1.0),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Silu => value / (1.0 + (-value).exp()),
            Self::Sqrt => value.sqrt(),
            Self::HardSigmoid { alpha, beta } => (value.mul_add(alpha, beta)).clamp(0.0, 1.0),
        }
    }
}

// Abramowitz-Stegun 7.1.26. The maximum absolute error is about 1.5e-7,
// which is below the accumulated F32 error of the surrounding GELU graph.
#[inline]
fn erf(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / x.mul_add(0.327_591_1, 1.0);
    let polynomial = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - polynomial * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        assert!(erf(0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.842_700_8).abs() < 2e-7);
        assert!((erf(-2.0) + 0.995_322_3).abs() < 2e-7);
    }

    #[test]
    fn vector_gelu_stays_close_to_scalar_formula() {
        let input = (0..1025)
            .map(|index| index as f32 * (16.0 / 1024.0) - 8.0)
            .collect::<Vec<_>>();
        let expected = input
            .iter()
            .map(|value| UnaryOperation::Gelu.apply(*value))
            .collect::<Vec<_>>();
        let mut actual = input;
        unary_in_place(&mut actual, UnaryOperation::Gelu);
        let maximum_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 2.0e-5, "maximum error: {maximum_error}");
    }

    #[test]
    fn packed_gemm_matches_row_major_gemm() {
        let rows = 28;
        let inner = 7;
        let columns = 19;
        let left = (0..rows * inner)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) / 11.0)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 9.0)
            .collect::<Vec<_>>();
        let bias = (0..rows)
            .map(|row| (row as f32 - 8.0) / 7.0)
            .collect::<Vec<_>>();
        let mut packed = Vec::with_capacity(left.len());
        for row_start in (0..rows).step_by(12) {
            let block_rows = (rows - row_start).min(12);
            for index in 0..inner {
                for row in 0..block_rows {
                    packed.push(left[(row_start + row) * inner + index]);
                }
            }
        }
        let mut expected = vec![0.0; rows * columns];
        let mut actual = vec![0.0; rows * columns];
        gemm(
            &mut expected,
            &left,
            &right,
            rows,
            inner,
            columns,
            Some(&bias),
        );
        gemm_packed_left(
            &mut actual,
            &packed,
            &right,
            rows,
            inner,
            columns,
            Some(&bias),
        );
        let maximum_error = expected
            .iter()
            .zip(&actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 2e-6, "maximum error: {maximum_error}");
    }

    #[test]
    fn exact_sparse_gemm_handles_dynamic_column_tails() {
        let rows = 8;
        let inner = 7;
        let row_offsets = [0, 3, 5];
        let indices = [0, 2, 6, 1, 5];
        let weights = [
            1.0, -0.5, 0.25, 2.0, 0.75, 1.5, -1.0, 0.5, -2.0, 0.125, 0.375, 1.25, 0.5, -0.75, 2.0,
            1.0, -1.5, 0.25, 0.625, -0.125,
        ];
        let bias = [0.5, -0.25, 1.0, -1.0, 0.75, 0.0, -0.5, 0.25];

        for columns in [1, 15, 16, 17, 31, 32, 33] {
            let right = (0..inner * columns)
                .map(|index| ((index * 13 % 37) as f32 - 18.0) / 11.0)
                .collect::<Vec<_>>();
            let mut actual = vec![0.0; rows * columns];
            gemm_sparse_packed_left(
                &mut actual,
                &right,
                rows,
                inner,
                columns,
                Some(&bias),
                &row_offsets,
                &indices,
                &weights,
                None,
            );

            let mut expected = vec![0.0; rows * columns];
            for block in 0..2 {
                for row in 0..4 {
                    for column in 0..columns {
                        let output_row = block * 4 + row;
                        let mut sum = bias[output_row];
                        for entry in row_offsets[block]..row_offsets[block + 1] {
                            sum = weights[entry * 4 + row]
                                .mul_add(right[indices[entry] as usize * columns + column], sum);
                        }
                        expected[output_row * columns + column] = sum;
                    }
                }
            }
            assert_eq!(actual, expected, "column count {columns}");
        }
    }

    #[test]
    fn depthwise_same_matches_scalar_reference() {
        for kernel in [3, 5, 7, 9] {
            for (height, width) in [
                (2, 3),
                (9, 7),
                (9, 8),
                (9, 9),
                (9, 15),
                (9, 16),
                (9, 17),
                (9, 31),
                (9, 32),
                (9, 33),
                (9, 37),
                (11, 41),
            ] {
                let input = (0..height * width)
                    .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
                    .collect::<Vec<_>>();
                let weights = (0..kernel * kernel)
                    .map(|index| ((index * 11 % 31) as f32 - 15.0) / 19.0)
                    .collect::<Vec<_>>();
                let bias = -0.375;
                let padding = kernel / 2;
                let mut expected = vec![0.0; input.len()];
                for y in 0..height {
                    for x in 0..width {
                        let mut sum = bias;
                        for kernel_y in 0..kernel {
                            let padded_y = y + kernel_y;
                            if padded_y < padding || padded_y - padding >= height {
                                continue;
                            }
                            for kernel_x in 0..kernel {
                                let padded_x = x + kernel_x;
                                if padded_x < padding || padded_x - padding >= width {
                                    continue;
                                }
                                sum = input[(padded_y - padding) * width + padded_x - padding]
                                    .mul_add(weights[kernel_y * kernel + kernel_x], sum);
                            }
                        }
                        expected[y * width + x] = sum;
                    }
                }

                let mut actual = vec![0.0; input.len()];
                depthwise_conv2d_same(&mut actual, &input, &weights, height, width, kernel, bias);
                let maximum_error = expected
                    .iter()
                    .zip(&actual)
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    maximum_error < 2e-5,
                    "kernel={kernel}, shape={height}x{width}, maximum error={maximum_error}"
                );
            }
        }
    }

    #[test]
    fn depthwise_stride2_matches_scalar_reference() {
        for (height, width) in [
            (2usize, 3usize),
            (9, 7),
            (9, 8),
            (9, 9),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (9, 20),
            (9, 31),
            (9, 32),
            (9, 33),
            (9, 34),
            (9, 37),
            (10, 38),
            (47, 92),
        ] {
            let input = (0..height * width)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
                .collect::<Vec<_>>();
            let weights = (0..9)
                .map(|index| ((index * 11 % 31) as f32 - 15.0) / 19.0)
                .collect::<Vec<_>>();
            let bias = -0.375;
            let output_height = height.div_ceil(2);
            let output_width = width.div_ceil(2);
            let mut expected = vec![0.0; output_height * output_width];
            for output_y in 0..output_height {
                for output_x in 0..output_width {
                    let mut sum = bias;
                    for kernel_y in 0..3 {
                        let input_y = output_y * 2 + kernel_y;
                        if input_y == 0 || input_y > height {
                            continue;
                        }
                        for kernel_x in 0..3 {
                            let input_x = output_x * 2 + kernel_x;
                            if input_x == 0 || input_x > width {
                                continue;
                            }
                            sum = input[(input_y - 1) * width + input_x - 1]
                                .mul_add(weights[kernel_y * 3 + kernel_x], sum);
                        }
                    }
                    expected[output_y * output_width + output_x] = sum;
                }
            }
            let mut actual = vec![0.0; expected.len()];
            depthwise_conv2d_same_3x3_stride2(&mut actual, &input, &weights, height, width, bias);
            assert_eq!(actual, expected, "shape={height}x{width}");
        }
    }

    #[test]
    fn softmax_is_normalized_and_ordered() {
        let input = [-3.0, 0.5, 2.0, -0.25, 1.0, 0.0, -1.0];
        let maximum = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut expected = input.map(|value| (value - maximum).exp());
        let sum = expected.iter().sum::<f32>();
        expected.iter_mut().for_each(|value| *value /= sum);
        let mut values = input;
        softmax_in_place(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 2e-6);
        assert!(
            values
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() < 2e-6)
        );
        assert_eq!(
            values
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index),
            Some(2)
        );
    }

    #[test]
    fn fused_bias_softmax_matches_separate_operations() {
        let mut expected = (0..37)
            .map(|index| ((index * 17 % 43) as f32 - 21.0) / 13.0)
            .collect::<Vec<_>>();
        let bias = (0..37)
            .map(|index| ((index * 11 % 31) as f32 - 15.0) / 19.0)
            .collect::<Vec<_>>();
        let mut actual = expected.clone();
        add_in_place(&mut expected, &bias);
        softmax_in_place(&mut expected);
        bias_softmax_in_place(&mut actual, &bias);
        assert_eq!(actual, expected);
    }

    #[test]
    fn fused_residual_mul_preserves_two_step_rounding() {
        let mut values = (0..37)
            .map(|index| (index as f32 - 19.0) / 7.0)
            .collect::<Vec<_>>();
        let mut expected = values.clone();
        for value in &mut expected {
            let original = *value;
            let scaled = original.mul_add(0.375, 0.0);
            *value = scaled.mul_add(1.0, original);
        }
        residual_mul_in_place(&mut values, 0.375);
        assert_eq!(values, expected);
    }

    #[test]
    fn relu_and_max_pool_ignore_a_single_nan() {
        let mut values = vec![-1.0; 17];
        values[0] = f32::NAN;
        unary_in_place(&mut values, UnaryOperation::Relu);
        assert_eq!(values[0], 0.0);

        let mut input = vec![1.0; 17];
        input[0] = f32::NAN;
        input[1] = 2.0;
        let mut output = vec![0.0; input.len()];
        max_pool_2x2_same_upper(&mut output, &input, 17);
        assert_eq!(output[0], 2.0);
    }
}
