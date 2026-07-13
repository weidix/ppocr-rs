# PP-OCRv6 Rust Inference Assessment

This repository contains a direct F32 Safetensors implementation of PP-OCRv6 medium, small, and
tiny detector/recognizer pairs using Candle. Its performance target is efficient native Metal
execution without routing inference through a separate system ML runtime. The experimental Burn
path uses ONNX only as an offline graph-import format; it is not part of the default runtime
dependency set.

## Reproducible Run

Download the exact Hugging Face revisions outside the repository:

```sh
hf download PaddlePaddle/PP-OCRv6_medium_det_safetensors model.safetensors \
  --revision 4236c2b61741a259c091fd879dcc4edc339e916c \
  --local-dir /tmp/ppocr-v6-models/det
hf download PaddlePaddle/PP-OCRv6_medium_rec_safetensors model.safetensors \
  --revision 024cad6a831de75c2c3c26e711ba8c4a82ccd24b \
  --local-dir /tmp/ppocr-v6-models/rec
hf download PaddlePaddle/PP-OCRv6_small_det_safetensors model.safetensors \
  --revision eae2ee920a39fb3087637d3dbb58df1896ec1f24 \
  --local-dir /tmp/ppocr-v6-models/small-det
hf download PaddlePaddle/PP-OCRv6_small_rec_safetensors model.safetensors \
  --revision fe049fb103f57443fe8840c54ed06b702f3c1de5 \
  --local-dir /tmp/ppocr-v6-models/small-rec
hf download PaddlePaddle/PP-OCRv6_tiny_det_safetensors model.safetensors \
  --revision 07595f982703daf0d4e120a12a01da8073542f3a \
  --local-dir /tmp/ppocr-v6-models/tiny-det
hf download PaddlePaddle/PP-OCRv6_tiny_rec_safetensors model.safetensors \
  --revision 6f2d2d51b4b4226d7a2329a02f416f4994106f3a \
  --local-dir /tmp/ppocr-v6-models/tiny-rec

cargo run --release -- \
  --det-model /tmp/ppocr-v6-models/det/model.safetensors \
  --rec-model /tmp/ppocr-v6-models/rec/model.safetensors \
  --image /path/to/validation/images/video0001_f00000060.jpg \
  --annotations /path/to/validation/annotations.jsonl \
  --device metal --warmup 1 --iterations 3
```

Use `--device cpu` for the CPU path. To run a smaller model, add matching size flags:

```sh
cargo run --release -- \
  --det-model /tmp/ppocr-v6-models/tiny-det/model.safetensors \
  --rec-model /tmp/ppocr-v6-models/tiny-rec/model.safetensors \
  --det-size tiny --rec-size tiny \
  --det-max-side 736 --rec-max-width 320 \
  --image /path/to/validation/images/video0001_f00000060.jpg \
  --annotations /path/to/validation/annotations.jsonl \
  --device metal --warmup 1 --iterations 5
```

`--det-max-side` and `--rec-max-width` are opt-in latency controls. `--det-max-side` only downsizes an image whose longest edge exceeds the limit. With neither flag, detector preprocessing preserves the released `limit_type=min` behavior and recognition keeps its dynamic width (up to 3200). The command prints JSON with model load, preprocessing, host-to-device transfer, synchronized forward, and transfer-plus-forward timing.

## Direct Safetensors Results

Hardware: Apple M4 (10-core GPU, 16 GB unified memory), macOS 26.5, Rust 1.92.0, Candle 0.10.2.
All values are milliseconds per item. `forward` includes the model's sigmoid/softmax but excludes JPEG decoding, preprocessing, DB box extraction, and CTC text decoding. Metal measurements call `Device::synchronize()` after every iteration.

| Model / input policy | Detector input | Detector Metal p50 | Recognizer input | Recognizer Metal p50 | Notes |
| --- | --- | ---: | --- | ---: | --- |
| medium, dynamic | `[1,3,1088,1920]` | 7254 ms | `[1,3,48,1047]` | 497 ms | Existing 3-run baseline on `f00000060` |
| small, dynamic | `[1,3,1088,1920]` | 1746.5 ms | `[1,3,48,1047]` | 179.1 ms | 3 runs after one warmup |
| tiny, dynamic | `[1,3,1088,1920]` | 863.9 ms | `[1,3,48,1047]` | 61.8 ms | 3 runs after one warmup |
| small, `--det-max-side 736 --rec-max-width 320` | `[1,3,416,736]` | 334.2 ms | `[1,3,48,320]` | 134.5 ms | 5 runs after one warmup |
| tiny, `--det-max-side 736 --rec-max-width 320` | `[1,3,416,736]` | 161.1 ms | `[1,3,48,320]` | 51.3 ms | 5 runs after one warmup |

At the constrained setting, tiny detector throughput is about 6.2 frames/s before preprocessing and DB postprocessing. One detector plus one 320-wide recognition crop is about 212 ms of forward work; real OCR latency grows with the number of detected crops.

Small/tiny graphs were checked against the official PaddleX dynamic Safetensors runner with the same weights. Recognition's first 12 CTC argmax values matched for both models. Detection output ranges matched; total output differed by below 0.1% because this project uses `image` triangle resize while the reference uses Pillow bilinear resize.

The recognition input now follows the released v6 processor: BGR `(pixel / 255 - 0.5) / 0.5`, with zero-filled right padding. Detector preprocessing retains BGR ImageNet normalization.

## Metal Limits

The medium detector's four 9x9 projection convolutions must be tiled to stay within Metal memory. A full-HD one-run sweep found 8 output rows per tile best:

| Rows per tile | Detector forward |
| ---: | ---: |
| 8 | 7089.35 ms |
| 16 | 7177.29 ms |
| 32 | 7314.90 ms |
| 64 | 7837.61 ms |

All four runs had identical output sums. F16 was also probed on tiny at `[1,3,416,736]`: it produced an all-zero detector mask and no material speed gain. The implementation deliberately remains F32-only.

## Burn Metal Probe

Burn is an opt-in runtime feature of this crate. It imports official ONNX once at build time,
embeds the generated graph and weights, and executes through Metal/WGPU with fusion. Inputs are
pre-uploaded, and the timed interval is `forward` plus `Metal::sync`; model loading,
preprocessing, and output readback are excluded.

Prepare fixed-shape models with the Rust utility, then build the Burn benchmark without enabling
the default Candle feature:

```sh
cargo run --release --no-default-features --features onnx-tools \
  --bin ppocr-onnx-staticize -- det.onnx /tmp/det-fixed.onnx --shape 1 3 416 736
cargo run --release --no-default-features --features onnx-tools \
  --bin ppocr-onnx-staticize -- rec.onnx /tmp/rec-fixed.onnx --shape 1 3 48 320

PPOCR_BURN_DET_ONNX=/tmp/det-fixed.onnx \
PPOCR_BURN_REC_ONNX=/tmp/rec-fixed.onnx \
cargo run --release --no-default-features --features burn-bench \
  --bin ppocr-burn-bench -- \
  --image /path/to/image.jpg --annotations /path/to/annotations.jsonl --warmup 5 --runs 30
```

`--annotations` is optional. The benchmark validates detector shape `[1,3,416,736]` and fixes
recognition to `[1,3,48,320]`; preprocessing and model loading are outside timed loops.

The benchmark vendors a narrow `burn-cubecl` patch: with autotune disabled, compatible
ungrouped 1x1 convolutions use Burn's existing im2col/matmul implementation; every other
convolution stays on the Direct path. This avoids the unstable global autotuner without changing
grouped, strided, padded, or non-1x1 convolution selection.

On the same M4, F32 official models at the constrained shapes produced:

| Model | Input | Samples | Burn Metal p50 | p90 | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| medium detector | `[1,3,416,736]` | 5 warmups, 20 runs | 167.100 ms | 167.709 ms | 3.31x faster than the 552.820 ms Direct control |
| medium recognizer | `[1,3,48,320]` | 5 warmups, 20 runs | 17.627 ms | 17.870 ms | 8.20x faster than the 144.493 ms Direct control |
| small detector | `[1,3,416,736]` | 5 warmups, 30 runs | 89.821 ms | 91.512 ms | Historical Direct baseline |
| small recognizer | `[1,3,48,320]` | 5 warmups, 30 runs | 29.205 ms | 29.802 ms | Historical Direct baseline |
| tiny detector | `[1,3,416,736]` | 5 warmups, 50 runs | 29.252 ms | 30.803 ms | Historical Direct baseline |
| tiny recognizer | `[1,3,48,320]` | 5 warmups, 50 runs | 7.260 ms | 7.488 ms | Historical Direct baseline |

Tiny is about 5.5x faster for detection and 7.1x faster for recognition than the current direct
F32 Candle measurements at the same shapes. Before timing, the tool performs a synchronized
forward/readback sanity check. The small-model smoke run produced detector output
`[1,1,416,736]` with sum `19076.196182` and recognizer output `[1,40,18710]` with sum
`40.000132`; all values were finite. This checks import and execution health, not end-to-end OCR
semantic parity.

For the medium run, the Direct and guarded-im2col output sums were `8150.233108` and
`8150.233923` for detection, and both `40.000226` for recognition. The existing synchronized
shape/finite/nonzero checks still pass. This is a numerical smoke check, not end-to-end OCR
semantic parity.

The default `metal + fusion` configuration uses the guarded 1x1 path above. Enabling Burn's
optional autotune feature is not usable on this macOS Metal/WGPU combination: its GPU-to-CPU
tuning-buffer map fails validation and the fusion scheduler subsequently panics. No autotuned
latency is reported.

### Batch Mode

Batch mode was evaluated on the prior Direct baseline and intentionally not retained. Small
detector throughput did not improve, medium detector throughput plateaued at batch 2 while still
slower than the CPU reference, and recognition's limited gains did not make it competitive. The
guarded 1x1 path delivers the material batch-1 gain; a future batch experiment must be measured
again against this optimized baseline.

## External ORT CPU / ANE Comparison

The following direct-inference measurements were supplied from the same local evaluation effort
and were produced by an external ORT benchmark. The first table is treated as recognition because
it reports lines per second; the second is treated as detection because it reports frames per
second. They are external CPU and ANE reference measurements, not Candle or Burn results and not
an ORT runtime dependency of this repository.

| Recognition size | CPU p50 | CPU mean | CPU throughput | ANE p50 | ANE mean | ANE throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 1.80 ms | 1.84 ms | 555 lines/s | 3.67 ms | 3.72 ms | 273 lines/s |
| small | 8.10 ms | 8.10 ms | 124 lines/s | 11.45 ms | 11.43 ms | 87 lines/s |
| medium | 21.38 ms | 21.41 ms | 47 lines/s | 30.52 ms | 30.55 ms | 33 lines/s |

| Detection size | CPU p50 | CPU mean | CPU throughput | ANE p50 | ANE mean | ANE throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 48.39 ms | 48.38 ms | 20.7 frames/s | 47.92 ms | 48.02 ms | 20.9 frames/s |
| small | 95.44 ms | 95.68 ms | 10.5 frames/s | 90.94 ms | 90.85 ms | 11.0 frames/s |
| medium | 517.06 ms | 519.91 ms | 1.9 frames/s | 354.48 ms | 360.46 ms | 2.8 frames/s |

Using the optimized medium Burn p50 values above, the direct latency comparison is:

| Recognition size | Burn p50 | Burn vs CPU | Burn vs ANE |
| --- | ---: | ---: | ---: |
| tiny | 7.260 ms | 4.03x slower | 1.98x slower |
| small | 29.205 ms | 3.61x slower | 2.55x slower |
| medium | 17.627 ms | 1.21x faster | 1.73x faster |

| Detection size | Burn p50 | Burn vs CPU | Burn vs ANE |
| --- | ---: | ---: | ---: |
| tiny | 29.252 ms | 39.5% faster | 39.0% faster |
| small | 89.821 ms | 5.9% faster | 1.2% faster |
| medium | 167.100 ms | 3.09x faster | 2.12x faster |

The optimized medium path has an advantage over both external references on this workload.
These derived comparisons are provisional until the two paths are confirmed to use identical model
revisions, input shapes, preprocessing, dtypes, warmups, synchronization, and timing boundaries.
Burn's measurements use pre-uploaded fixed tensors and synchronized `forward + Metal::sync` only.

## Native CPU ONNX Control

RTen 0.24 is a separate pure-Rust CPU control using the official matching ONNX repositories, not a conversion performed in this assessment and not a replacement for direct Safetensors loading. On the same Apple M4 with four performance cores, one warmup plus one synchronized real-image run gave:

| Sample / model | Input | RTen forward | Other measured work |
| --- | --- | ---: | --- |
| `f00022200` det | `[1,3,1088,1920]` | 1501.82 ms | JPEG decode 4.58 ms; resize plus BGR normalization 17.25 ms |
| GT crop from `f00022200` rec | `[1,3,48,320]` | 36.99 ms | Preprocessing 0.068 ms |

The RTen control is the practical CPU option found in this assessment. It remains too slow for real-time full-HD detection, but is roughly an order of magnitude faster than the direct Candle CPU detector and recognition paths.

## Conclusion

The direct Candle path now loads and executes all official PP-OCRv6 size tiers. Medium remains unsuitable for local real-time full-HD detection. Tiny is the practical ceiling within the current direct F32/Candle implementation: about 1.2 frames/s at dynamic full-HD input, or 6.2 frames/s with a 736-pixel maximum side before postprocessing. The 736 setting trades small-text recall for latency and needs task-level accuracy evaluation before deployment.

The medium detector is roughly 451 GMAC at the validation-frame input size. Candle's generic implementation expands 9x9 convolutions through im2col and evaluates grouped convolutions as per-group chunks. Tiling avoids an approximately 10 GB temporary Metal buffer, but custom fused/grouped convolution kernels would be required to materially change the medium model's speed limit.

## Runtime Options

| Runtime | Model format | CPU | GPU | Assessment |
| --- | --- | --- | --- | --- |
| Candle 0.10.2 | Requested Safetensors | Yes | Metal/CUDA | Implemented for medium/small/tiny. Tiny is usable for reduced-resolution local inference; medium needs custom fused/grouped convolution kernels for a materially higher ceiling. |
| Burn 0.21 with WGPU/CubeCL | Official ONNX imported at build time | Yes | Metal/WGPU/CUDA backends | Native Metal p50 at constrained shapes: optimized medium 167.100/17.627 ms (detector/recognizer). A vendored guarded 1x1 im2col fallback replaces Direct only where shape-safe; autotune remains unstable on this host. |
| RTen 0.24 | Official matching ONNX | Yes | No | Measured on a real validation frame: 1502 ms detector and 37 ms for a 320-wide crop. Strong pure-Rust CPU fallback, but it does not read the supplied Safetensors directly. |
| Wonnx 0.5 | Official ONNX | No practical result | WGPU/Metal | Current model preparation fails on unsupported HardSigmoid; detector also needs ConvTranspose support. |
| Tract 0.23 | Official ONNX | Yes | No | Current dynamic PP-OCRv6 ONNX optimization fails at the first convolution. |

Neither path invokes a separate system ML runtime.

## Compatibility Notes

- Candle 0.11.0 does not compile on stable Rust 1.92 for this Apple Silicon host because its NEON `f16` code uses unstable `stdarch_neon_f16`; this project intentionally pins Candle 0.10.2.
- Small detector/recognizer use `--det-size small --rec-size small`; tiny uses `--det-size tiny --rec-size tiny`. Tiny recognition emits 6906 logits and needs the released `ppocrv6_tiny_dict` for text decoding. This benchmark executable intentionally stops at CTC logits.
- Validation annotations contain polygons and detector scores but no text transcription. They validate frame/crop latency and detector output shape, not end-to-end OCR accuracy.
