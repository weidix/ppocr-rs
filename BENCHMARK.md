# PP-OCRv6 Rust Inference Assessment

This repository contains a direct F32 Safetensors implementation of PP-OCRv6 medium, small, and
tiny detector/recognizer pairs using Candle. Its performance target is efficient native Metal
execution without routing inference through a separate system ML runtime. The experimental Burn
path uses ONNX only as an offline graph-import format; it is not part of the default runtime
dependency set.

## Reproducible Run

Download and verify the pinned Hugging Face revisions into the repository-local `models/`
directory:

```sh
./scripts/download-models.sh

cargo run --release -- \
  --det-model models/medium-det/model.safetensors \
  --rec-model models/medium-rec/model.safetensors \
  --image /path/to/validation/images/video0001_f00000060.jpg \
  --dict models/medium-rec/inference.yml \
  --det-max-side 736 \
  --device metal --output /tmp/candle-ocr.json
```

Use `--device cpu` for the CPU path. To run a smaller model, add matching size flags:

```sh
cargo run --release -- \
  --det-model models/tiny-det/model.safetensors \
  --rec-model models/tiny-rec/model.safetensors \
  --det-size tiny --rec-size tiny \
  --det-max-side 736 --rec-max-width 320 \
  --image /path/to/validation/images/video0001_f00000060.jpg \
  --dict models/tiny-rec/inference.yml \
  --device metal --output /tmp/candle-ocr.json
```

`--det-max-side` and `--rec-max-width` are opt-in latency controls. `--det-max-side` only downsizes an image whose longest edge exceeds the limit. With neither flag, detector preprocessing preserves the released `limit_type=min` behavior and recognition keeps its dynamic width (up to 3200). The runtime performs detector-mask postprocessing, polygon rectification, overlapping wide-line splitting, and CTC greedy decoding without annotations, then writes reading-order OCR JSON.

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

## Burn End-to-End OCR

`ppocr-burn` runs the fixed-shape Burn models as a complete OCR pipeline: aspect-preserving
detector letterboxing, detector-mask postprocessing, rotated text crops, fixed-width recognition,
CTC greedy decoding, and JSON output. `--dict` accepts a one-entry-per-line dictionary or the
released recognizer `inference.yml`, whose `character_dict` matches the model class count. The
standard implicit PaddleX space class is handled automatically when present.

```sh
PPOCR_BURN_DET_ONNX=/tmp/det-fixed.onnx \
PPOCR_BURN_REC_ONNX=/tmp/rec-fixed.onnx \
cargo run --release --no-default-features --features burn-infer \
  --bin ppocr-burn -- \
  --image /path/to/image.jpg --dict /path/to/inference.yml --output /tmp/ocr.json
```

The runtime defaults to binary threshold `0.2`, box-score threshold `0.4`, unclip ratio `1.4`,
and at most 1,000 boxes. All values can be adjusted through `--help`. The bounded fixed inputs
avoid dynamic-shape compilation behavior, and the runtime synchronizes before every output
readback.

For long lines, build a wider fixed recognizer and pass its width to `ppocr-burn`; the command
line width must exactly match the ONNX shape used during Burn model generation:

```sh
cargo run --release --no-default-features --features onnx-tools \
  --bin ppocr-onnx-staticize -- rec.onnx /tmp/rec-fixed-1024.onnx --shape 1 3 48 1024

PPOCR_BURN_DET_ONNX=/tmp/det-fixed.onnx \
PPOCR_BURN_REC_ONNX=/tmp/rec-fixed-1024.onnx \
cargo run --release --no-default-features --features burn-infer \
  --bin ppocr-burn -- \
  --image /path/to/image.jpg --dict /path/to/inference.yml --rec-width 1024 --output /tmp/ocr.json
```

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
`ppocr-burn-bench` therefore must be built with a recognizer staticized at width `320`; use
`ppocr-burn --rec-width N` for wide recognizer models.

The benchmark vendors a narrow `burn-cubecl` patch: with autotune disabled, compatible
ungrouped 1x1 convolutions use Burn's existing im2col/matmul implementation; every other
convolution stays on the Direct path. This avoids the unstable global autotuner without changing
grouped, strided, padded, or non-1x1 convolution selection.

On the same M4, F32 official models were freshly measured with five warmups and 30 timed runs at
the constrained shapes:

| Model | Input | Configuration | Burn Metal p50 | p90 | Throughput at p50 |
| --- | --- | --- | ---: | ---: | ---: |
| medium detector | `[1,3,416,736]` | guarded 1x1 im2col | 166.927 ms | 167.613 ms | 5.99 frames/s |
| medium recognizer | `[1,3,48,320]` | guarded 1x1 im2col | 17.634 ms | 17.752 ms | 56.71 lines/s |
| small detector | `[1,3,416,736]` | guarded 1x1 im2col | 47.855 ms | 48.590 ms | 20.90 frames/s |
| small recognizer | `[1,3,48,320]` | guarded 1x1 im2col | 10.278 ms | 11.470 ms | 97.30 lines/s |
| tiny detector | `[1,3,416,736]` | guarded 1x1 im2col | 28.606 ms | 29.214 ms | 34.96 frames/s |
| tiny recognizer | `[1,3,48,320]` | guarded 1x1 im2col | 3.918 ms | 5.142 ms | 255.23 lines/s |

Before timing, the tool performs a synchronized forward/readback sanity check. All six fresh runs
returned finite, nonzero outputs with the expected detector shape `[1,1,416,736]`; recognizer
outputs had shape `[1,40,18710]` for medium/small and `[1,40,6906]` for tiny. This checks import
and execution health, not end-to-end OCR semantic parity.

The default `metal + fusion` configuration uses the guarded 1x1 path above. Enabling Burn's
optional autotune feature is not usable on this macOS Metal/WGPU combination: its GPU-to-CPU
tuning-buffer map fails validation and the fusion scheduler subsequently panics. No autotuned
latency is reported.

## Direct Safetensors WGPU Probe

The root package's `gpu` feature executes the released F32 Safetensors directly. It uses
Metal exclusively on macOS and Vulkan exclusively on other supported platforms. Model loading,
input upload, and output readback are excluded from timing; each sample records the fixed-shape
graph, submits it, and waits for completion. These Apple M4 measurements use the same fixed raw
F32 inputs as the ORT probe, with five warmups and 30 timed runs.

| Model | WGPU p50 | p90 | Throughput | Burn p50 | WGPU vs Burn | ORT CPU / GPU / ANE p50 | WGPU vs ORT GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium detector | 173.197 ms | 174.520 ms | 5.77 frames/s | 166.927 ms | 3.8% slower | 286.663 / 184.75 / 180.73 ms | 6.3% faster |
| medium recognizer | 19.050 ms | 20.177 ms | 52.49 lines/s | 17.634 ms | 8.0% slower | 21.208 / 34.35 / 31.87 ms | 44.5% faster |
| small detector | 36.589 ms | 36.778 ms | 27.33 frames/s | 47.855 ms | 23.5% faster | 52.489 / 55.58 / 52.87 ms | 34.2% faster |
| small recognizer | 8.843 ms | 8.945 ms | 113.08 lines/s | 10.278 ms | 14.0% faster | 8.184 / 13.35 / 11.91 ms | 33.8% faster |
| tiny detector | 20.337 ms | 21.575 ms | 49.17 frames/s | 28.606 ms | 28.9% faster | 26.704 / 29.65 / 27.40 ms | 31.4% faster |
| tiny recognizer | 3.837 ms | 3.870 ms | 260.61 lines/s | 3.918 ms | 2.1% faster | 1.941 / 4.34 / 4.52 ms | 11.6% faster |

WGPU beats Burn for both small and tiny models. Medium is within 3.8% for detection and 8.0% for
recognition, so the current implementation does not claim a clean Burn win at that size. All six
WGPU results beat the documented ORT GPU row. That ORT row is CoreML `CPUAndGPU`, which permits CPU
fallback and is not a pure Metal baseline. ORT CPU remains faster for the small and tiny
recognizers.

The synchronized validation forward produced finite, nonzero output at the expected shape for all
six models. Detector comparison against ORT stayed below `3.734604e-7` maximum absolute error.
Medium, small, and tiny recognizers matched ORT argmax at `40/40` time steps. Tiny's larger
per-logit ORT difference is also reproduced by the independent CPU Safetensors implementation;
against the strict Safetensors reference, GPU maximum absolute error is about `4.83e-6`.

## Converted ONNX CPU Probe

The self-contained `cpu_onnx` runtime was measured using converted fixed-shape ONNX models on the
same M4 host. Each model uses one worker thread, five warmups, and 30 timed runs; model loading,
input generation, and output validation are outside the timed interval.

| Model | Input | cpu_onnx p50 | p90 | Throughput at p50 |
| --- | --- | ---: | ---: | ---: |
| medium detector | `[1,3,416,736]` | 972.947 ms | 977.228 ms | 1.03 frames/s |
| medium recognizer | `[1,3,48,320]` | 107.621 ms | 107.805 ms | 9.29 lines/s |
| small detector | `[1,3,416,736]` | 126.527 ms | 127.257 ms | 7.90 frames/s |
| small recognizer | `[1,3,48,320]` | 25.390 ms | 25.624 ms | 39.39 lines/s |
| tiny detector | `[1,3,416,736]` | 52.462 ms | 53.110 ms | 19.06 frames/s |
| tiny recognizer | `[1,3,48,320]` | 4.858 ms | 5.149 ms | 205.85 lines/s |

## Direct Safetensors CPU Probe

The direct `cpu` runtime was measured from the official Safetensors weights on the same M4 host.
Each model uses one worker thread, five warmups, and 30 timed runs; model loading, input
generation, and output validation are outside the timed interval.

| Model | Input | cpu p50 | p90 | Throughput at p50 |
| --- | --- | ---: | ---: | ---: |
| medium detector | `[1,3,416,736]` | 178.041 ms | 178.850 ms | 5.62 frames/s |
| medium recognizer | `[1,3,48,320]` | 18.754 ms | 18.971 ms | 53.32 lines/s |
| small detector | `[1,3,416,736]` | 44.782 ms | 45.091 ms | 22.33 frames/s |
| small recognizer | `[1,3,48,320]` | 7.038 ms | 7.226 ms | 142.09 lines/s |
| tiny detector | `[1,3,416,736]` | 24.944 ms | 25.826 ms | 40.09 frames/s |
| tiny recognizer | `[1,3,48,320]` | 1.943 ms | 2.092 ms | 514.67 lines/s |

These single-thread results beat the ORT CPU reference for five models and match tiny recognition
within `0.002 ms` (about `0.1%`).

The CPU runtime evaluates every nonzero convolution weight. Its sparse representation skips only
blocks that are exactly zero. On macOS, dense pointwise and tiled spatial convolutions use
Accelerate SGEMM, and Linear consumes `[out,in]` weights directly through a transposed SGEMM instead
of materializing two matrix transposes. AArch64 also has a dedicated exact-order 3x3 stride-two
depthwise kernel. Sparse kernels handle dynamic-width tail columns directly, so they never
reinterpret four-row packed weights as the twelve-row dense layout.

The optimized runtime was compared with the preceding exact-weight implementation using the same
fixed deterministic inputs. Medium/small/tiny detector maximum absolute differences were
`2.271e-6`, `3.185e-6`, and `1.26e-7`, with zero output-mask changes at threshold `0.5`.
Recognizer maximum absolute differences were `4.8101e-5`, `2.6226e-5`, and `1.02043e-4`; all three
matched the preceding implementation at all `40/40` argmax time steps. No weights are pruned,
quantized, or approximated.

The real validation crops containing `98tang.net` and `shtfab@gmail.com` were also run at their
dynamic widths of 1,596 and 2,052 pixels. Thirty consecutive native CPU runs matched the Candle
text exactly; steady-state p50 latency was `79.626 ms` and `101.398 ms`, respectively.

## RTen CPU ONNX Control

RTen 0.24 is a separate pure-Rust CPU control using the official matching ONNX repositories, not a
conversion performed in this assessment and not a replacement for direct Safetensors loading. Each
model was measured with one worker thread, five warmups, and 30 timed runs. RTen receives one
fixed randomly generated F32 input per model, reused for all runs.

Install `rten-cli` 0.24, then run the following commands for each medium, small, and tiny official
ONNX repository. `-n 35` performs five warmups followed by 30 samples; p50 and p90 are calculated
from samples 6 through 35. RTen runs the model only: model loading, image decoding, preprocessing,
and OCR postprocessing are outside the measurement.

```sh
RTEN=/path/to/rten

# Detector: fixed [1,3,416,736] random F32 input, one worker thread.
$RTEN /path/to/PP-OCRv6_tiny_det_onnx/inference.onnx \
  -t 1 -n 35 \
  -s '"DynamicDimension.1"=416' \
  -s '"DynamicDimension.2"=736'

# Recognizer: fixed [1,3,48,320] random F32 input, one worker thread.
$RTEN /path/to/PP-OCRv6_tiny_rec_onnx/inference.onnx \
  -t 1 -n 35 \
  -s '"DynamicDimension.1"=320'
```

| Model | Input | RTen p50 | p90 | Throughput at p50 |
| --- | --- | ---: | ---: | ---: |
| medium detector | `[1,3,416,736]` | 650.640 ms | 651.560 ms | 1.54 frames/s |
| medium recognizer | `[1,3,48,320]` | 110.970 ms | 111.530 ms | 9.01 lines/s |
| small detector | `[1,3,416,736]` | 107.970 ms | 108.350 ms | 9.26 frames/s |
| small recognizer | `[1,3,48,320]` | 24.570 ms | 24.700 ms | 40.70 lines/s |
| tiny detector | `[1,3,416,736]` | 45.460 ms | 46.970 ms | 22.00 frames/s |
| tiny recognizer | `[1,3,48,320]` | 4.630 ms | 4.910 ms | 215.98 lines/s |

## ONNX Runtime Fixed-Shape Probe

ONNX Runtime (ORT) CPU was measured with fixed F32 inputs: detector `[1,3,416,736]` and recognizer
`[1,3,48,320]`, one intra-op and one inter-op thread, five warmups, and 30 timed runs. GPU and ANE
rows retain their existing measurements. Values are milliseconds per item. Throughput is calculated
as `1000 / p50_ms`.

| Model | Input | Backend | ORT p50 | p90 | Throughput at p50 |
| --- | --- | --- | ---: | ---: | ---: |
| medium detector | `[1,3,416,736]` | CPU | 286.663 ms | 289.826 ms | 3.49 frames/s |
| medium detector | `[1,3,416,736]` | GPU | 184.75 ms | 211.84 ms | 5.41 frames/s |
| medium detector | `[1,3,416,736]` | ANE | 180.73 ms | 194.42 ms | 5.53 frames/s |
| medium recognizer | `[1,3,48,320]` | CPU | 21.208 ms | 21.505 ms | 47.15 lines/s |
| medium recognizer | `[1,3,48,320]` | GPU | 34.35 ms | 37.56 ms | 29.1 lines/s |
| medium recognizer | `[1,3,48,320]` | ANE | 31.87 ms | 32.73 ms | 31.4 lines/s |
| small detector | `[1,3,416,736]` | CPU | 52.489 ms | 53.190 ms | 19.05 frames/s |
| small detector | `[1,3,416,736]` | GPU | 55.58 ms | 56.15 ms | 18.0 frames/s |
| small detector | `[1,3,416,736]` | ANE | 52.87 ms | 54.39 ms | 18.9 frames/s |
| small recognizer | `[1,3,48,320]` | CPU | 8.184 ms | 8.415 ms | 122.19 lines/s |
| small recognizer | `[1,3,48,320]` | GPU | 13.35 ms | 13.73 ms | 74.9 lines/s |
| small recognizer | `[1,3,48,320]` | ANE | 11.91 ms | 12.15 ms | 84.0 lines/s |
| tiny detector | `[1,3,416,736]` | CPU | 26.704 ms | 27.389 ms | 37.45 frames/s |
| tiny detector | `[1,3,416,736]` | GPU | 29.65 ms | 30.29 ms | 33.7 frames/s |
| tiny detector | `[1,3,416,736]` | ANE | 27.40 ms | 28.19 ms | 36.5 frames/s |
| tiny recognizer | `[1,3,48,320]` | CPU | 1.941 ms | 2.418 ms | 515.20 lines/s |
| tiny recognizer | `[1,3,48,320]` | GPU | 4.34 ms | 4.58 ms | 230.5 lines/s |
| tiny recognizer | `[1,3,48,320]` | ANE | 4.52 ms | 5.21 ms | 221.2 lines/s |

## Conclusion

The direct Candle path now loads and executes all official PP-OCRv6 size tiers. Medium remains unsuitable for local real-time full-HD detection. Tiny is the practical ceiling within the current direct F32/Candle implementation: about 1.2 frames/s at dynamic full-HD input, or 6.2 frames/s with a 736-pixel maximum side before postprocessing. The 736 setting trades small-text recall for latency and needs task-level accuracy evaluation before deployment.

The medium detector is roughly 451 GMAC at the validation-frame input size. Candle's generic implementation expands 9x9 convolutions through im2col and evaluates grouped convolutions as per-group chunks. Tiling avoids an approximately 10 GB temporary Metal buffer, but custom fused/grouped convolution kernels would be required to materially change the medium model's speed limit.

## Runtime Options

| Runtime | Model format | CPU | GPU | Assessment |
| --- | --- | --- | --- | --- |
| Direct WGPU | Requested Safetensors | No | Metal/Vulkan | Fixed-shape medium/small/tiny detector and recognizer graphs. WGPU beats the ORT GPU baseline for all six; small/tiny beat Burn, while medium is within 8%. |
| Candle 0.10.2 | Requested Safetensors | Yes | Metal/CUDA | End-to-end OCR is implemented for medium/small/tiny. Tiny is usable for reduced-resolution local inference; medium needs custom fused/grouped convolution kernels for a materially higher ceiling. |
| cpu_onnx | Converted fixed-shape ONNX | Yes | No | Single-thread p50 detector/recognizer latency (ms): medium 972.947/107.621, small 126.527/25.390, tiny 52.462/4.858. |
| cpu | Requested Safetensors | Yes | No | Exact-weight single-thread p50 detector/recognizer latency (ms): medium 178.041/18.754, small 44.782/7.038, tiny 24.944/1.943. |
| Burn 0.21 with WGPU/CubeCL | Official ONNX imported at build time | Yes | Metal/WGPU/CUDA backends | Fresh guarded-path p50 detector/recognizer latency (ms): medium 166.927/17.634, small 47.855/10.278, tiny 28.606/3.918. |
| RTen 0.24 | Official matching ONNX | Yes | No | Single-thread p50 detector/recognizer latency (ms): medium 650.640/110.970, small 107.970/24.570, tiny 45.460/4.630. It does not read the supplied Safetensors directly. |
| ONNX Runtime (ORT) | ONNX | Yes | GPU/ANE | Fixed-shape p50 detector/recognizer latency (ms): single-thread CPU medium 286.663/21.208, small 52.489/8.184, tiny 26.704/1.941; GPU medium 184.75/34.35, small 55.58/13.35, tiny 29.65/4.34; ANE medium 180.73/31.87, small 52.87/11.91, tiny 27.40/4.52. |
| Wonnx 0.5 | Official ONNX | No practical result | WGPU/Metal | Current model preparation fails on unsupported HardSigmoid; detector also needs ConvTranspose support. |
| Tract 0.23 | Official ONNX | Yes | No | Current dynamic PP-OCRv6 ONNX optimization fails at the first convolution. |

Neither path invokes a separate system ML runtime.

## Compatibility Notes

- Candle 0.11.0 does not compile on stable Rust 1.92 for this Apple Silicon host because its NEON `f16` code uses unstable `stdarch_neon_f16`; this project intentionally pins Candle 0.10.2.
- Small detector/recognizer use `--det-size small --rec-size small`; tiny uses `--det-size tiny --rec-size tiny`. Tiny recognition emits 6906 logits and needs the released `ppocrv6_tiny_dict` for text decoding. The Candle CLI decodes those logits into full OCR output.
- Validation annotations contain polygons and detector scores but no text transcription. They validate frame/crop latency and detector output shape, not end-to-end OCR accuracy.
