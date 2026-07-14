# ppocr-rs

Unified Rust crate for PP-OCRv6 inference, optional Burn Metal OCR and benchmark binaries, and the
Candle CRNN trainer. Runtime dependencies are selected explicitly through Cargo features.

| Feature | Purpose | Default |
| --- | --- | --- |
| `candle-metal` | PP-OCRv6 Safetensors inference on Metal | Yes |
| `candle` | PP-OCRv6 Safetensors inference on CPU | No |
| `cpu` | Direct Safetensors inference with native CPU kernels | No |
| `gpu` | Direct Safetensors inference through WGPU (Metal/Vulkan) | No |
| `cpu_onnx` | Self-contained SIMD CPU inference for converted ONNX models | No |
| `cpu_onnx-convert` | Offline ONNX to `.ppocr-cpu` conversion | No |
| `burn-infer` | Fixed-shape ONNX Burn Metal end-to-end OCR | No |
| `burn-bench` | Fixed-shape ONNX Burn Metal benchmark | No |
| `onnx-tools` | Rust ONNX fixed-shape utility | No |
| `training` | Candle CRNN trainer | No |

## Safetensors GPU Backend

The `gpu` feature runs the official medium, small, and tiny detector and recognizer models directly
from F32 Safetensors. It selects Metal on macOS and Vulkan on other platforms.

```sh
cargo run --release --no-default-features --features gpu --bin ppocr-gpu-bench -- \
  --model det --size tiny --weights /path/to/model.safetensors \
  --height 416 --width 736 --warmup 5 --runs 30
```

The Rust API is available under `ppocr_rs::gpu`.

## Safetensors CPU Backend

The `cpu` feature provides direct PP-OCRv6 Safetensors CPU inference for the medium, small, and tiny
detector and recognizer models from the PaddlePaddle PP-OCRv6 collection.

```sh
./scripts/download-models.sh

cargo run --release --no-default-features --features cpu --bin ppocr-cpu-bench -- \
  --kind det --size tiny \
  --height 416 --width 736 --threads 4 --warmup 5 --runs 30
```

The benchmark defaults to the pinned local weight at
`models/<medium|small|tiny>-<det|rec>/model.safetensors`. Revisions, sizes, and SHA256 values are
recorded in `models.json`; pass `--model` only to override the local model explicitly.

The Rust API is available under `ppocr_rs::cpu`.

The native CPU path evaluates every nonzero model weight. On macOS, large pointwise convolutions
use Accelerate SGEMM; the custom sparse kernels skip only exact-zero blocks and support arbitrary
recognizer widths without a packed-layout fallback.

## Converted ONNX CPU Backend

The `cpu_onnx` runtime implements the PP-OCRv6 operators locally and does not invoke ONNX Runtime,
Burn, Candle, RTen, BLAS, or another inference library. Its only compute dependency is Rayon for
the fixed worker pool. `rten-onnx` is enabled only by `cpu_onnx-convert` to decode ONNX offline and
is not linked into deployed `cpu_onnx` builds. Direct Safetensors inference uses the `cpu` feature
above.

Convert a model at its deployed fixed shape, then benchmark the packed model:

```sh
cargo run --release --no-default-features --features cpu_onnx-convert \
  --bin ppocr-cpu-onnx-convert -- \
  inference.onnx model.ppocr-cpu --shape 1 3 48 320

cargo run --release --no-default-features --features cpu_onnx \
  --bin ppocr-cpu-onnx-bench -- \
  model.ppocr-cpu --threads 4 --warmup 5 --runs 30
```

The runtime API accepts a contiguous NCHW F32 tensor:

```rust
use ppocr_rs::cpu_onnx::{CpuModel, CpuOptions, Tensor};

let model = CpuModel::load("model.ppocr-cpu", CpuOptions { threads: 4 })?;
let input = Tensor::from_f32(model.input_shape().to_vec(), input_values)?;
let output = model.run(input)?;
```

On the Apple M4 benchmark host, the optimized tiny detector runs at 21.56 ms p50 versus the
documented 27.84 ms ORT baseline. Repeated tiny recognizer runs measure 1.83-1.91 ms p50 versus
1.81 ms for ORT. The recognizer comparison used identical input: all 40 argmax positions matched,
with maximum absolute output error `8.24e-5`.

Run end-to-end Candle OCR. `--dict` accepts either one character per line or the matching PaddleX
recognizer `inference.yml`:

```sh
cargo run --release -- \
  --det-model /path/to/det/model.safetensors \
  --rec-model /path/to/rec/model.safetensors \
  --image /path/to/image.jpg \
  --dict /path/to/PP-OCRv6_medium_rec_onnx/inference.yml \
  --det-max-side 736 --output /tmp/ocr.json
```

The Candle output is JSON with reading-order text boxes, original-image quadrilaterals, detector
scores, decoded text, and CTC confidence. It resizes the detector input to 32-pixel alignment,
rectifies each detected text polygon, and splits wide text crops with overlap before recognition.

Create Burn-compatible fixed-shape ONNX models without Python:

```sh
cargo run --release --no-default-features --features onnx-tools \
  --bin ppocr-onnx-staticize -- input.onnx output.onnx --shape 1 3 416 736
```

Run end-to-end Burn OCR after setting both fixed-model paths. `--dict` accepts either a one-entry-
per-line dictionary or the released recognizer `inference.yml`, which contains the matching
character list. When a released PaddleX config omits its standard trailing space entry, the
decoder recognizes that extra output class automatically:

```sh
PPOCR_BURN_DET_ONNX=/tmp/det-fixed.onnx \
PPOCR_BURN_REC_ONNX=/tmp/rec-fixed.onnx \
cargo run --release --no-default-features --features burn-infer \
  --bin ppocr-burn -- \
  --image /path/to/image.jpg --dict /path/to/inference.yml --output /tmp/ocr.json
```

The output is JSON with reading-order text boxes, their original-image quadrilaterals, detector
scores, decoded text, and CTC confidence. The fixed detector uses aspect-preserving resize plus
right/bottom padding to `[1,3,416,736]`; the default recognizer uses `[1,3,48,320]` for every
detected crop. Crops wider than the fixed recognizer aspect ratio are split with overlap and
decoded in reading order instead of being horizontally squashed.

For wider lines, staticize the recognizer at the desired width and pass the same width to
`ppocr-burn`. The `--rec-width` value must exactly match the recognizer ONNX shape used when
building the Burn binary:

```sh
cargo run --release --no-default-features --features onnx-tools \
  --bin ppocr-onnx-staticize -- rec.onnx /tmp/rec-fixed-1024.onnx --shape 1 3 48 1024

PPOCR_BURN_DET_ONNX=/tmp/det-fixed.onnx \
PPOCR_BURN_REC_ONNX=/tmp/rec-fixed-1024.onnx \
cargo run --release --no-default-features --features burn-infer \
  --bin ppocr-burn -- \
  --image /path/to/image.jpg --dict /path/to/inference.yml --rec-width 1024 --output /tmp/ocr.json
```

Run the Burn benchmark after setting the same fixed-model paths:

```sh
PPOCR_BURN_DET_ONNX=/tmp/det-fixed.onnx \
PPOCR_BURN_REC_ONNX=/tmp/rec-fixed.onnx \
cargo run --release --no-default-features --features burn-bench \
  --bin ppocr-burn-bench -- --help
```

Run the migrated Candle trainer:

```sh
cargo run --release --no-default-features --features training \
  --bin ppocr-candle-train -- --help
```
