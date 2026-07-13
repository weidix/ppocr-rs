# PP-OCR Burn Benchmark

Standalone pure-Rust macOS Metal benchmark for fixed-shape PP-OCRv6 ONNX models. It uses Burn `0.21.0` with `metal` and `fusion`; this default configuration is the supported stable path. Its image loading, fixed-shape preprocessing, and optional annotation crop are implemented inside this tool, with no dependency on the parent project or another inference engine.

Staticize dynamic models first (requires `pip install onnx`):

```sh
python3 tools/ppocr-burn-bench/staticize_onnx.py det.onnx /tmp/det-fixed.onnx --shape 1 3 416 736
python3 tools/ppocr-burn-bench/staticize_onnx.py rec.onnx /tmp/rec-fixed.onnx --shape 1 3 48 320
```

Build and run from the repository root:

```sh
PPOCR_BURN_DET_ONNX=/tmp/det-fixed.onnx \
PPOCR_BURN_REC_ONNX=/tmp/rec-fixed.onnx \
cargo run --release --manifest-path tools/ppocr-burn-bench/Cargo.toml -- \
  --image /path/to/image.jpg --annotations /path/to/annotations.jsonl --warmup 5 --runs 30
```

`--annotations` is optional; without it recognition uses the whole image. The image path must naturally produce detector shape `[1,3,416,736]` under the standard `max_side=736` preprocessing rule; the tool validates this rather than resizing the source image to force a benchmark shape. Recognition is fixed at `[1,3,48,320]`. `--warmup` must be at least `1`. Preprocessing, model loading, CTC decoding, detection postprocessing, and image I/O are outside the timed loops. Each timed iteration calls `Metal::sync`, so results include queued GPU execution.

Before timing, the tool runs one synchronized forward pass per model, reads both outputs, and checks the imported model's generated output shape, finite values, and nonzero total. This is an output sanity check only; it does not establish OCR semantic parity.

The optional `autotune` feature is intentionally not enabled by default. On this macOS 26.5.2 / WGPU 29 / Burn 0.21 combination it fails during the first tuning pass with an invalid GPU-to-CPU map buffer, followed by CubeCL and Burn fusion `CallError` panics; it is not a supported benchmark configuration.

The imported graph is embedded into the benchmark binary at build time. This is an experiment, not a runtime backend: it requires static input shapes, macOS Metal, and ONNX operations supported by Burn's importer. Generated Rust and weights live only under Cargo's build output; no model binary is stored in this repository.
