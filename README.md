# ppocr-rs

Unified Rust crate for PP-OCRv6 inference, optional Burn Metal OCR and benchmark binaries, and the
Candle CRNN trainer. Runtime dependencies are selected explicitly through Cargo features.

| Feature | Purpose | Default |
| --- | --- | --- |
| `candle-metal` | PP-OCRv6 Safetensors inference on Metal | Yes |
| `candle` | PP-OCRv6 Safetensors inference on CPU | No |
| `burn-infer` | Fixed-shape ONNX Burn Metal end-to-end OCR | No |
| `burn-bench` | Fixed-shape ONNX Burn Metal benchmark | No |
| `onnx-tools` | Rust ONNX fixed-shape utility | No |
| `training` | Candle CRNN trainer | No |

Run the default Candle binary:

```sh
cargo run --release -- --help
```

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
right/bottom padding to `[1,3,416,736]`; the recognizer uses `[1,3,48,320]` for every detected
crop. Crops wider than the fixed recognizer aspect ratio are split with overlap and decoded in
reading order instead of being horizontally squashed.

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
