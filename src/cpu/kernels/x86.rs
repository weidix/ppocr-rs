//! x86-64 AVX2 and FMA kernels.

use core::arch::x86_64::*;

// Every entry point in this module requires the caller to have checked both
// AVX2 and FMA at runtime. Loads and stores are unaligned and only cover full
// vectors; each kernel handles its remaining elements with safe slice access.

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn axpy(output: &mut [f32], input: &[f32], scale: f32) {
    debug_assert_eq!(output.len(), input.len());
    let scale = _mm256_set1_ps(scale);
    let vector_len = output.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is inside both equal-length slices.
        let (input, output_value) = unsafe {
            (
                _mm256_loadu_ps(input.as_ptr().add(offset)),
                _mm256_loadu_ps(output.as_ptr().add(offset)),
            )
        };
        let result = _mm256_fmadd_ps(input, scale, output_value);
        // SAFETY: The same full-vector bound applies to the output store.
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(offset), result) };
    }
    let scale = _mm256_cvtss_f32(scale);
    for index in vector_len..output.len() {
        output[index] = input[index].mul_add(scale, output[index]);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn mul_in_place(output: &mut [f32], input: &[f32]) {
    debug_assert_eq!(output.len(), input.len());
    let vector_len = output.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is inside both equal-length slices.
        let (left, right) = unsafe {
            (
                _mm256_loadu_ps(output.as_ptr().add(offset)),
                _mm256_loadu_ps(input.as_ptr().add(offset)),
            )
        };
        // SAFETY: The store covers the same in-bounds output vector.
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(offset), _mm256_mul_ps(left, right)) };
    }
    for index in vector_len..output.len() {
        output[index] *= input[index];
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn affine(values: &mut [f32], scale: f32, bias: f32) {
    let scale_vector = _mm256_set1_ps(scale);
    let bias_vector = _mm256_set1_ps(bias);
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let result = _mm256_fmadd_ps(value, scale_vector, bias_vector);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), result) };
    }
    for value in &mut values[vector_len..] {
        *value = value.mul_add(scale, bias);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn residual_mul(values: &mut [f32], gate: f32) {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let gate_vector = _mm256_set1_ps(gate);
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let original = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let scaled = _mm256_fmadd_ps(original, gate_vector, zero);
        let output = _mm256_fmadd_ps(scaled, one, original);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), output) };
    }
    for value in &mut values[vector_len..] {
        let original = *value;
        let scaled = original.mul_add(gate, 0.0);
        *value = scaled.mul_add(1.0, original);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn square(values: &mut [f32]) {
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), _mm256_mul_ps(value, value)) };
    }
    for value in &mut values[vector_len..] {
        *value *= *value;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn relu(values: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let vector_len = values.len() / 8 * 8;
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // Passing zero as the second operand also matches `f32::max` for NaN.
        let result = _mm256_max_ps(value, zero);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), result) };
    }
    for value in &mut values[vector_len..] {
        *value = value.max(0.0);
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_rows_8<const ROWS: usize, const PACKED_LEFT: bool>(
    output: &mut [f32],
    left: &[f32],
    right: &[f32],
    inner: usize,
    columns: usize,
    right_stride: usize,
    bias: Option<&[f32]>,
    column_bias: Option<&[f32]>,
) {
    debug_assert!(ROWS > 0 && ROWS <= 12);
    debug_assert_eq!(output.len(), ROWS * columns);
    debug_assert_eq!(left.len(), ROWS * inner);
    debug_assert!(inner == 0 || right.len() >= inner.saturating_sub(1) * right_stride + columns);
    debug_assert!(bias.is_none_or(|bias| bias.len() == ROWS));
    debug_assert!(column_bias.is_none_or(|bias| bias.len() == columns));
    debug_assert!(bias.is_none() || column_bias.is_none());
    debug_assert!(!PACKED_LEFT || column_bias.is_none());

    let vector_columns = columns / 8 * 8;
    for column in (0..vector_columns).step_by(8) {
        let mut accumulators = [_mm256_setzero_ps(); ROWS];
        if let Some(column_bias) = column_bias {
            // SAFETY: A full vector is available because `column < vector_columns`.
            let initial = unsafe { _mm256_loadu_ps(column_bias.as_ptr().add(column)) };
            accumulators.fill(initial);
        } else {
            for row in 0..ROWS {
                accumulators[row] = _mm256_set1_ps(bias.map_or(0.0, |bias| bias[row]));
            }
        }

        for index in 0..inner {
            // SAFETY: Matrix dimensions and `right_stride` were validated above.
            let right_vector =
                unsafe { _mm256_loadu_ps(right.as_ptr().add(index * right_stride + column)) };
            for (row, accumulator) in accumulators.iter_mut().enumerate() {
                let left_index = if PACKED_LEFT {
                    index * ROWS + row
                } else {
                    row * inner + index
                };
                // SAFETY: Both supported layouts contain exactly `ROWS * inner` values.
                let scale = _mm256_set1_ps(unsafe { *left.get_unchecked(left_index) });
                *accumulator = _mm256_fmadd_ps(scale, right_vector, *accumulator);
            }
        }

        for (row, accumulator) in accumulators.iter().copied().enumerate() {
            // SAFETY: Each output row has `columns` values and this is a full vector.
            unsafe {
                _mm256_storeu_ps(output.as_mut_ptr().add(row * columns + column), accumulator)
            };
        }
    }

    for row in 0..ROWS {
        for column in vector_columns..columns {
            let mut sum =
                column_bias.map_or_else(|| bias.map_or(0.0, |bias| bias[row]), |bias| bias[column]);
            for index in 0..inner {
                let left_index = if PACKED_LEFT {
                    index * ROWS + row
                } else {
                    row * inner + index
                };
                sum = left[left_index].mul_add(right[index * right_stride + column], sum);
            }
            output[row * columns + column] = sum;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn gelu(values: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let inv_sqrt_two = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
    let sign_bit = _mm256_set1_ps(-0.0);
    let vector_len = values.len() / 8 * 8;

    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let input = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let scaled = _mm256_mul_ps(input, inv_sqrt_two);
        let absolute = _mm256_andnot_ps(sign_bit, scaled);
        let denominator = _mm256_fmadd_ps(absolute, _mm256_set1_ps(0.327_591_1), one);
        let t = reciprocal(denominator);
        let mut polynomial =
            _mm256_fmadd_ps(t, _mm256_set1_ps(1.061_405_4), _mm256_set1_ps(-1.453_152_1));
        polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(1.421_413_8));
        polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(-0.284_496_72));
        polynomial = _mm256_fmadd_ps(polynomial, t, _mm256_set1_ps(0.254_829_6));
        polynomial = _mm256_mul_ps(polynomial, t);
        let exponential = exp256(_mm256_sub_ps(zero, _mm256_mul_ps(absolute, absolute)));
        let positive_erf = _mm256_fnmadd_ps(polynomial, exponential, one);
        let negative_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(scaled, zero);
        let erf = _mm256_blendv_ps(
            positive_erf,
            _mm256_sub_ps(zero, positive_erf),
            negative_mask,
        );
        let output = _mm256_mul_ps(_mm256_mul_ps(half, input), _mm256_add_ps(one, erf));
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), output) };
    }

    for value in &mut values[vector_len..] {
        *value = scalar_gelu(*value);
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn softmax(values: &mut [f32]) {
    let vector_len = values.len() / 8 * 8;
    let mut maxima = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let maximum = &mut maxima[vector & 3];
        // Value is first so a NaN lane leaves the previous maximum intact.
        *maximum = _mm256_max_ps(value, *maximum);
    }
    let mut maximum_lanes = [f32::NEG_INFINITY; 32];
    for (vector, maximum) in maxima.into_iter().enumerate() {
        // SAFETY: Each store writes eight values into its own array segment.
        unsafe { _mm256_storeu_ps(maximum_lanes.as_mut_ptr().add(vector * 8), maximum) };
    }
    let mut maximum = maximum_lanes.into_iter().fold(f32::NEG_INFINITY, f32::max);
    for &value in &values[vector_len..] {
        maximum = maximum.max(value);
    }

    let maximum_vector = _mm256_set1_ps(maximum);
    let mut sums = [_mm256_setzero_ps(); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let exponential = exp256(_mm256_sub_ps(value, maximum_vector));
        let sum = &mut sums[vector & 3];
        *sum = _mm256_add_ps(*sum, exponential);
        // SAFETY: The store covers the same in-bounds vector.
        unsafe { _mm256_storeu_ps(values.as_mut_ptr().add(offset), exponential) };
    }
    let mut sum_lanes = [0.0; 32];
    for (vector, sum) in sums.into_iter().enumerate() {
        // SAFETY: Each store writes eight values into its own array segment.
        unsafe { _mm256_storeu_ps(sum_lanes.as_mut_ptr().add(vector * 8), sum) };
    }
    let mut sum = sum_lanes.into_iter().sum::<f32>();
    for value in &mut values[vector_len..] {
        *value = (*value - maximum).exp();
        sum += *value;
    }

    let reciprocal = _mm256_set1_ps(sum.recip());
    for offset in (0..vector_len).step_by(8) {
        // SAFETY: `offset..offset + 8` is a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        // SAFETY: The store covers the same in-bounds vector.
        unsafe {
            _mm256_storeu_ps(
                values.as_mut_ptr().add(offset),
                _mm256_mul_ps(value, reciprocal),
            )
        };
    }
    let reciprocal = _mm256_cvtss_f32(reciprocal);
    for value in &mut values[vector_len..] {
        *value *= reciprocal;
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn sum(values: &[f32]) -> f32 {
    let vector_len = values.len() / 8 * 8;
    let mut sums = [_mm256_setzero_ps(); 4];
    for (vector, offset) in (0..vector_len).step_by(8).enumerate() {
        // SAFETY: Each offset points at a complete in-bounds vector.
        let value = unsafe { _mm256_loadu_ps(values.as_ptr().add(offset)) };
        let sum = &mut sums[vector & 3];
        *sum = _mm256_add_ps(*sum, value);
    }
    let mut lanes = [0.0; 32];
    for (vector, sum) in sums.into_iter().enumerate() {
        // SAFETY: Each store writes eight values into its own array segment.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr().add(vector * 8), sum) };
    }
    lanes.into_iter().sum::<f32>() + values[vector_len..].iter().sum::<f32>()
}

#[target_feature(enable = "avx2,fma")]
fn reciprocal(value: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0);
    let mut estimate = _mm256_rcp_ps(value);
    estimate = _mm256_mul_ps(estimate, _mm256_fnmadd_ps(value, estimate, two));
    _mm256_mul_ps(estimate, _mm256_fnmadd_ps(value, estimate, two))
}

#[target_feature(enable = "avx2,fma")]
fn exp256(value: __m256) -> __m256 {
    let value = _mm256_max_ps(
        _mm256_set1_ps(-87.0),
        _mm256_min_ps(_mm256_set1_ps(87.0), value),
    );
    let exponent = _mm256_cvtps_epi32(_mm256_mul_ps(
        value,
        _mm256_set1_ps(std::f32::consts::LOG2_E),
    ));
    let remainder = _mm256_fnmadd_ps(
        _mm256_cvtepi32_ps(exponent),
        _mm256_set1_ps(std::f32::consts::LN_2),
        value,
    );
    let mut polynomial = _mm256_set1_ps(1.0 / 120.0);
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0 / 24.0));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0 / 6.0));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(0.5));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0));
    polynomial = _mm256_fmadd_ps(polynomial, remainder, _mm256_set1_ps(1.0));
    let exponent_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(exponent, _mm256_set1_epi32(127)));
    _mm256_mul_ps(polynomial, _mm256_castsi256_ps(exponent_bits))
}

#[inline]
fn scalar_gelu(input: f32) -> f32 {
    let x = input * std::f32::consts::FRAC_1_SQRT_2;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / x.mul_add(0.327_591_1, 1.0);
    let polynomial = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let erf = sign * (1.0 - polynomial * (-x * x).exp());
    0.5 * input * (1.0 + erf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simd_available() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len());
        let maximum_error = expected
            .iter()
            .zip(actual)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            maximum_error <= tolerance,
            "maximum error {maximum_error} exceeded {tolerance}"
        );
    }

    #[test]
    fn elementwise_kernels_match_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let input = (0..39)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 7.0)
            .collect::<Vec<_>>();
        let base = (0..39)
            .map(|index| ((index * 17 % 37) as f32 - 18.0) / 11.0)
            .collect::<Vec<_>>();

        let mut expected = base.clone();
        expected
            .iter_mut()
            .zip(&input)
            .for_each(|(output, input)| *output = input.mul_add(-0.37, *output));
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { axpy(&mut actual, &input, -0.37) };
        assert_close(&expected, &actual, 2e-7);

        let mut expected = base.clone();
        expected
            .iter_mut()
            .zip(&input)
            .for_each(|(output, input)| *output *= input);
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { mul_in_place(&mut actual, &input) };
        assert_close(&expected, &actual, 0.0);

        let mut expected = base.clone();
        expected
            .iter_mut()
            .for_each(|value| *value = value.mul_add(1.27, -0.13));
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { affine(&mut actual, 1.27, -0.13) };
        assert_close(&expected, &actual, 2e-7);

        let mut expected = base.clone();
        expected.iter_mut().for_each(|value| *value *= *value);
        let mut actual = base.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { square(&mut actual) };
        assert_close(&expected, &actual, 0.0);
    }

    #[test]
    fn activation_kernels_match_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let input = (0..43)
            .map(|index| ((index * 19 % 47) as f32 - 23.0) / 8.0)
            .collect::<Vec<_>>();

        let expected = input.iter().map(|value| value.max(0.0)).collect::<Vec<_>>();
        let mut actual = input.clone();
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { relu(&mut actual) };
        assert_close(&expected, &actual, 0.0);

        let expected = input
            .iter()
            .map(|&value| scalar_gelu(value))
            .collect::<Vec<_>>();
        let mut actual = input;
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { gelu(&mut actual) };
        assert_close(&expected, &actual, 3e-6);
    }

    #[test]
    fn softmax_matches_scalar_with_tail() {
        if !simd_available() {
            return;
        }
        let mut actual = (0..37)
            .map(|index| ((index * 23 % 41) as f32 - 20.0) / 6.0)
            .collect::<Vec<_>>();
        let mut expected = actual.clone();
        let maximum = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = expected
            .iter_mut()
            .map(|value| {
                *value = (*value - maximum).exp();
                *value
            })
            .sum::<f32>();
        expected.iter_mut().for_each(|value| *value /= sum);

        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe { softmax(&mut actual) };
        assert_close(&expected, &actual, 3e-6);
        assert!((actual.iter().sum::<f32>() - 1.0).abs() < 2e-6);
    }

    #[test]
    fn gemm_kernels_match_scalar_with_column_tail() {
        if !simd_available() {
            return;
        }
        const ROWS: usize = 8;
        let inner = 7;
        let columns = 19;
        let left = (0..ROWS * inner)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) / 11.0)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|index| ((index * 13 % 31) as f32 - 15.0) / 9.0)
            .collect::<Vec<_>>();
        let column_bias = (0..columns)
            .map(|column| (column as f32 - 8.0) / 7.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; ROWS * columns];
        for row in 0..ROWS {
            for column in 0..columns {
                let mut sum = column_bias[column];
                for index in 0..inner {
                    sum = left[row * inner + index].mul_add(right[index * columns + column], sum);
                }
                expected[row * columns + column] = sum;
            }
        }
        let mut actual = vec![0.0; expected.len()];
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe {
            gemm_rows_8::<ROWS, false>(
                &mut actual,
                &left,
                &right,
                inner,
                columns,
                columns,
                None,
                Some(&column_bias),
            )
        };
        assert_close(&expected, &actual, 2e-6);
    }

    #[test]
    fn packed_gemm_kernel_matches_scalar_with_column_tail() {
        if !simd_available() {
            return;
        }
        const ROWS: usize = 12;
        let inner = 5;
        let columns = 17;
        let row_major = (0..ROWS * inner)
            .map(|index| ((index * 11 % 37) as f32 - 18.0) / 12.0)
            .collect::<Vec<_>>();
        let mut packed = Vec::with_capacity(row_major.len());
        for index in 0..inner {
            for row in 0..ROWS {
                packed.push(row_major[row * inner + index]);
            }
        }
        let right = (0..inner * columns)
            .map(|index| ((index * 7 % 23) as f32 - 11.0) / 8.0)
            .collect::<Vec<_>>();
        let bias = (0..ROWS)
            .map(|row| (row as f32 - 5.0) / 9.0)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; ROWS * columns];
        for row in 0..ROWS {
            for column in 0..columns {
                let mut sum = bias[row];
                for index in 0..inner {
                    sum = row_major[row * inner + index]
                        .mul_add(right[index * columns + column], sum);
                }
                expected[row * columns + column] = sum;
            }
        }
        let mut actual = vec![0.0; expected.len()];
        // SAFETY: The runtime feature check above covers this AVX2+FMA kernel.
        unsafe {
            gemm_rows_8::<ROWS, true>(
                &mut actual,
                &packed,
                &right,
                inner,
                columns,
                columns,
                Some(&bias),
                None,
            )
        };
        assert_close(&expected, &actual, 2e-6);
    }
}
