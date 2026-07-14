//! GPU runtime errors.

use safetensors::Dtype;
use std::{fmt, io, path::PathBuf};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    ReadFile {
        path: PathBuf,
        source: io::Error,
    },
    Safetensors(safetensors::SafeTensorError),
    TensorNotFound {
        name: String,
    },
    UnsupportedDtype {
        name: String,
        found: Dtype,
    },
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    TensorSizeOverflow {
        name: String,
        shape: Vec<usize>,
    },
    TensorByteLength {
        name: String,
        expected: usize,
        found: usize,
    },
    InvalidTensorRange {
        name: String,
        start: usize,
        end: usize,
        file_len: usize,
    },
    InvalidModel(String),
    InvalidInput(String),
    Gpu(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadFile { path, source } => {
                write!(formatter, "read weights {}: {source}", path.display())
            }
            Self::Safetensors(source) => write!(formatter, "decode safetensors: {source}"),
            Self::TensorNotFound { name } => write!(formatter, "tensor {name:?} was not found"),
            Self::UnsupportedDtype { name, found } => {
                write!(formatter, "tensor {name:?} has dtype {found}; expected F32")
            }
            Self::ShapeMismatch {
                name,
                expected,
                found,
            } => write!(
                formatter,
                "tensor {name:?} has shape {found:?}; expected {expected:?}"
            ),
            Self::TensorSizeOverflow { name, shape } => write!(
                formatter,
                "tensor {name:?} shape {shape:?} overflows the platform address space"
            ),
            Self::TensorByteLength {
                name,
                expected,
                found,
            } => write!(
                formatter,
                "tensor {name:?} contains {found} bytes; expected {expected} bytes for F32"
            ),
            Self::InvalidTensorRange {
                name,
                start,
                end,
                file_len,
            } => write!(
                formatter,
                "tensor {name:?} byte range {start}..{end} is outside a {file_len}-byte file"
            ),
            Self::InvalidModel(message) => write!(formatter, "invalid model: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid input: {message}"),
            Self::Gpu(message) => write!(formatter, "GPU error: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReadFile { source, .. } => Some(source),
            Self::Safetensors(source) => Some(source),
            _ => None,
        }
    }
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(source: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(source)
    }
}
