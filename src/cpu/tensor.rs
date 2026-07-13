use anyhow::{Result, bail};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: TensorData,
}

#[derive(Clone, Debug)]
pub(crate) enum TensorData {
    F32(Arc<Vec<f32>>),
    I64(Arc<Vec<i64>>),
}

impl Tensor {
    pub fn from_f32(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        validate_len(&shape, data.len())?;
        Ok(Self {
            shape,
            data: TensorData::F32(Arc::new(data)),
        })
    }

    pub fn from_i64(shape: impl Into<Vec<usize>>, data: Vec<i64>) -> Result<Self> {
        let shape = shape.into();
        validate_len(&shape, data.len())?;
        Ok(Self {
            shape,
            data: TensorData::I64(Arc::new(data)),
        })
    }

    pub(crate) fn new_f32(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(element_count(&shape), Some(data.len()));
        Self {
            shape,
            data: TensorData::F32(Arc::new(data)),
        }
    }

    pub(crate) fn new_i64(shape: Vec<usize>, data: Vec<i64>) -> Self {
        debug_assert_eq!(element_count(&shape), Some(data.len()));
        Self {
            shape,
            data: TensorData::I64(Arc::new(data)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        match &self.data {
            TensorData::F32(data) => data.len(),
            TensorData::I64(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.data {
            TensorData::F32(data) => Ok(data),
            TensorData::I64(_) => bail!("expected an f32 tensor"),
        }
    }

    pub fn as_i64(&self) -> Result<&[i64]> {
        match &self.data {
            TensorData::I64(data) => Ok(data),
            TensorData::F32(_) => bail!("expected an i64 tensor"),
        }
    }

    pub(crate) fn into_f32(self) -> Result<Vec<f32>> {
        match self.data {
            TensorData::F32(data) => {
                Ok(Arc::try_unwrap(data).unwrap_or_else(|data| (*data).clone()))
            }
            TensorData::I64(_) => bail!("expected an f32 tensor"),
        }
    }

    pub(crate) fn into_i64(self) -> Result<Vec<i64>> {
        match self.data {
            TensorData::I64(data) => {
                Ok(Arc::try_unwrap(data).unwrap_or_else(|data| (*data).clone()))
            }
            TensorData::F32(_) => bail!("expected an i64 tensor"),
        }
    }
}

pub(crate) fn element_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |size, dim| size.checked_mul(*dim))
}

pub(crate) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![1; shape.len()];
    for index in (1..shape.len()).rev() {
        result[index - 1] = result[index] * shape[index];
    }
    result
}

fn validate_len(shape: &[usize], len: usize) -> Result<()> {
    let expected = element_count(shape).ok_or_else(|| anyhow::anyhow!("tensor shape overflow"))?;
    if expected != len {
        bail!("tensor shape {shape:?} requires {expected} values, found {len}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_tensor_length() {
        assert!(Tensor::from_f32([2, 3], vec![0.0; 6]).is_ok());
        assert!(Tensor::from_f32([2, 3], vec![0.0; 5]).is_err());
    }

    #[test]
    fn computes_contiguous_strides() {
        assert_eq!(strides(&[2, 3, 4]), [12, 4, 1]);
    }
}
