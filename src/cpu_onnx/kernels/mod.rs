use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
mod neon;

#[inline]
pub(crate) fn fill(values: &mut [f32], value: f32) {
    values.fill(value);
}

#[inline]
pub(crate) fn axpy(output: &mut [f32], input: &[f32], scale: f32) {
    debug_assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The implementation only performs unaligned loads/stores within
        // the bounds of equally sized slices. NEON is mandatory on AArch64.
        unsafe { neon::axpy(output, input, scale) };
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

pub(crate) fn mul_in_place(output: &mut [f32], input: &[f32]) {
    debug_assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Equal-length slices bound every vector load and store.
        unsafe { neon::mul_in_place(output, input) };
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
    #[cfg(not(target_arch = "aarch64"))]
    for value in values {
        *value = value.mul_add(scale, bias);
    }
}

pub(crate) fn square_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: The in-place kernel only accesses the supplied slice.
        unsafe { neon::square(values) };
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
    gemm_impl(
        output, left, right, rows, inner, columns, bias, None, false, None, false,
    );
}

pub(crate) fn gemm_gelu(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
) {
    gemm_impl(
        output,
        left,
        right,
        rows,
        inner,
        columns,
        bias,
        None,
        false,
        Some(UnaryOperation::Gelu),
        false,
    );
}

pub(crate) fn gemm_packed_left(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
) {
    gemm_impl(
        output, left, right, rows, inner, columns, bias, None, true, None, false,
    );
}

pub(crate) fn gemm_packed_left_gelu(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    rows: usize,
    inner: usize,
    columns: usize,
    bias: Option<&[f32]>,
) {
    gemm_impl(
        output,
        left,
        right,
        rows,
        inner,
        columns,
        bias,
        None,
        true,
        Some(UnaryOperation::Gelu),
        false,
    );
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
    debug_assert_eq!(output.len(), rows * columns);
    debug_assert_eq!(left.len(), rows * inner);
    debug_assert_eq!(right.len(), inner * columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == rows));
    debug_assert!(column_bias.is_none_or(|bias| bias.len() == columns));
    debug_assert!(bias.is_none() || column_bias.is_none());
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
                );
                if row_softmax {
                    for row in output.chunks_mut(columns) {
                        softmax_in_place(row);
                    }
                } else if let Some(activation) = activation {
                    unary_chunk(output, activation);
                }
            }
        });
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
) {
    #[cfg(target_arch = "aarch64")]
    if rows == 12 && packed_left {
        // SAFETY: The caller supplies exactly twelve complete output rows and
        // twelve interleaved weights for every inner-dimension position.
        debug_assert!(column_bias.is_none());
        unsafe { neon::gemm_12x8_packed(output, left, right, inner, columns, right_stride, bias) };
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
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if rows == 4 {
        // SAFETY: Same bounds argument as the eight-row kernel.
        unsafe {
            if packed_left {
                debug_assert!(column_bias.is_none());
                neon::gemm_4x16_packed(output, left, right, inner, columns, right_stride, bias)
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
}

pub(crate) fn max_pool_2x2_same_upper(output: &mut [f32], input: &[f32], width: usize) {
    debug_assert_eq!(output.len(), input.len());
    debug_assert!(width > 0 && input.len().is_multiple_of(width));
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

pub(crate) fn mean(values: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    let sum = {
        // SAFETY: The kernel only loads complete vectors within the slice and
        // handles its tail through safe indexing.
        unsafe { neon::sum(values) }
    };
    #[cfg(not(target_arch = "aarch64"))]
    let sum = values.iter().copied().sum::<f32>();
    sum / values.len() as f32
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
    fn apply(self, value: f32) -> f32 {
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
    fn softmax_is_normalized_and_ordered() {
        let mut values = [-3.0, 0.5, 2.0, -0.25, 1.0, 0.0, -1.0];
        softmax_in_place(&mut values);
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 2e-6);
        assert_eq!(
            values
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index),
            Some(2)
        );
    }
}
