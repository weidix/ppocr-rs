use super::error::{Error, Result};
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, fs, ops::Range, path::Path};

const F32_BYTES: usize = size_of::<f32>();

#[derive(Debug)]
pub struct Weights {
    bytes: Vec<u8>,
    tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Debug)]
struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    range: Range<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct F32Tensor<'a> {
    name: &'a str,
    shape: &'a [usize],
    bytes: &'a [u8],
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| Error::ReadFile {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let (header_len, metadata) = SafeTensors::read_metadata(&bytes)?;
        let data_start = size_of::<u64>()
            .checked_add(header_len)
            .ok_or(Error::Safetensors(
                safetensors::SafeTensorError::InvalidHeaderLength,
            ))?;
        let mut tensors = BTreeMap::new();

        for name in metadata.offset_keys() {
            let info = metadata
                .info(&name)
                .ok_or_else(|| Error::TensorNotFound { name: name.clone() })?;
            let start = data_start.checked_add(info.data_offsets.0).ok_or_else(|| {
                Error::InvalidTensorRange {
                    name: name.clone(),
                    start: info.data_offsets.0,
                    end: info.data_offsets.1,
                    file_len: bytes.len(),
                }
            })?;
            let end = data_start.checked_add(info.data_offsets.1).ok_or_else(|| {
                Error::InvalidTensorRange {
                    name: name.clone(),
                    start,
                    end: info.data_offsets.1,
                    file_len: bytes.len(),
                }
            })?;
            if start > end || end > bytes.len() {
                return Err(Error::InvalidTensorRange {
                    name,
                    start,
                    end,
                    file_len: bytes.len(),
                });
            }

            tensors.insert(
                name,
                TensorInfo {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    range: start..end,
                },
            );
        }

        Ok(Self { bytes, tensors })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor(&self, name: &str) -> Result<F32Tensor<'_>> {
        let (name, info) =
            self.tensors
                .get_key_value(name)
                .ok_or_else(|| Error::TensorNotFound {
                    name: name.to_owned(),
                })?;
        if info.dtype != Dtype::F32 {
            return Err(Error::UnsupportedDtype {
                name: name.clone(),
                found: info.dtype,
            });
        }

        let element_count = element_count(name, &info.shape)?;
        let expected_bytes =
            element_count
                .checked_mul(F32_BYTES)
                .ok_or_else(|| Error::TensorSizeOverflow {
                    name: name.clone(),
                    shape: info.shape.clone(),
                })?;
        let bytes =
            self.bytes
                .get(info.range.clone())
                .ok_or_else(|| Error::InvalidTensorRange {
                    name: name.clone(),
                    start: info.range.start,
                    end: info.range.end,
                    file_len: self.bytes.len(),
                })?;
        if bytes.len() != expected_bytes {
            return Err(Error::TensorByteLength {
                name: name.clone(),
                expected: expected_bytes,
                found: bytes.len(),
            });
        }

        Ok(F32Tensor {
            name,
            shape: &info.shape,
            bytes,
        })
    }

    pub fn tensor_with_shape(&self, name: &str, expected: &[usize]) -> Result<F32Tensor<'_>> {
        let tensor = self.tensor(name)?;
        tensor.expect_shape(expected)?;
        Ok(tensor)
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(self.tensor(name)?.shape())
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        Ok(self.tensor(name)?.bytes())
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.tensor(name)?.to_f32()
    }
}

impl<'a> F32Tensor<'a> {
    pub fn name(self) -> &'a str {
        self.name
    }

    pub fn shape(self) -> &'a [usize] {
        self.shape
    }

    pub fn len(self) -> usize {
        self.bytes.len() / F32_BYTES
    }

    pub fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }

    pub fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    pub fn expect_shape(self, expected: &[usize]) -> Result<Self> {
        if self.shape != expected {
            return Err(Error::ShapeMismatch {
                name: self.name.to_owned(),
                expected: expected.to_vec(),
                found: self.shape.to_vec(),
            });
        }
        Ok(self)
    }

    pub fn to_f32(self) -> Result<Vec<f32>> {
        let mut chunks = self.bytes.chunks_exact(F32_BYTES);
        let values = chunks
            .by_ref()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        if !chunks.remainder().is_empty() || values.len() != self.len() {
            return Err(Error::TensorByteLength {
                name: self.name.to_owned(),
                expected: self.len().saturating_mul(F32_BYTES),
                found: self.bytes.len(),
            });
        }
        Ok(values)
    }
}

fn element_count(name: &str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| Error::TensorSizeOverflow {
            name: name.to_owned(),
            shape: shape.to_vec(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, serialize, tensor::TensorView};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn archive(dtype: Dtype, shape: Vec<usize>, bytes: &[u8]) -> Vec<u8> {
        let tensor = TensorView::new(dtype, shape, bytes).unwrap();
        serialize([("model.weight", tensor)], None).unwrap()
    }

    #[test]
    fn reads_f32_names_shapes_bytes_and_values() {
        let raw = f32_bytes(&[1.25, -2.5, 0.0, f32::INFINITY]);
        let weights = Weights::from_bytes(archive(Dtype::F32, vec![2, 2], &raw)).unwrap();

        assert_eq!(weights.len(), 1);
        assert!(!weights.is_empty());
        assert!(weights.contains("model.weight"));
        assert_eq!(weights.names().collect::<Vec<_>>(), ["model.weight"]);

        let tensor = weights.tensor_with_shape("model.weight", &[2, 2]).unwrap();
        assert_eq!(tensor.name(), "model.weight");
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.bytes(), raw);
        assert_eq!(tensor.to_f32().unwrap(), [1.25, -2.5, 0.0, f32::INFINITY]);
    }

    #[test]
    fn rejects_missing_tensor_wrong_shape_and_non_f32_dtype() {
        let raw = f32_bytes(&[1.0, 2.0]);
        let weights = Weights::from_bytes(archive(Dtype::F32, vec![2], &raw)).unwrap();
        assert!(matches!(
            weights.tensor("missing"),
            Err(Error::TensorNotFound { .. })
        ));
        assert!(matches!(
            weights.tensor_with_shape("model.weight", &[1, 2]),
            Err(Error::ShapeMismatch { .. })
        ));

        let i32_raw = [1_i32.to_le_bytes(), 2_i32.to_le_bytes()].concat();
        let weights = Weights::from_bytes(archive(Dtype::I32, vec![2], &i32_raw)).unwrap();
        assert!(matches!(
            weights.tensor("model.weight"),
            Err(Error::UnsupportedDtype {
                found: Dtype::I32,
                ..
            })
        ));
    }

    #[test]
    fn rejects_truncated_safetensors() {
        let raw = f32_bytes(&[1.0, 2.0]);
        let mut bytes = archive(Dtype::F32, vec![2], &raw);
        bytes.pop();
        assert!(matches!(
            Weights::from_bytes(bytes),
            Err(Error::Safetensors(_))
        ));
    }

    #[test]
    fn loads_with_std_fs_read() {
        let raw = f32_bytes(&[3.0, 4.0]);
        let bytes = archive(Dtype::F32, vec![2], &raw);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ppocr-gpu-weights-{}-{nonce}.safetensors",
            std::process::id()
        ));
        fs::write(&path, bytes).unwrap();

        let weights = Weights::load(&path).unwrap();
        let values = weights.tensor_f32("model.weight").unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(values, [3.0, 4.0]);
    }
}
