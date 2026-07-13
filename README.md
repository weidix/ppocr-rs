# ppocr-rs

Unified Rust crate for PP-OCRv6 inference, the optional Burn Metal benchmark, and the Candle CRNN
trainer. Runtime dependencies are selected explicitly through Cargo features.

| Feature | Purpose | Default |
| --- | --- | --- |
| `candle-metal` | PP-OCRv6 Safetensors inference on Metal | Yes |
| `candle` | PP-OCRv6 Safetensors inference on CPU | No |
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

Run the Burn benchmark after setting both fixed-model paths:

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
