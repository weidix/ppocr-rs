# PP-OCRv6 Benchmarks

This document records direct Safetensors inference measurements for the native
CPU and WGPU backends. Model download, checksum validation, model loading,
input generation or upload, and output validation are outside the timed
interval.

## Reproduction

The benchmark commands resolve the pinned model package automatically. On a
new machine, the first command downloads it; use `--offline` to require an
existing cache and `--verify-models` to force a full checksum pass.

```sh
cargo run --release --bin ppocr-cpu-bench -- \
  --kind det --size tiny --height 416 --width 736 \
  --threads 4 --warmup 5 --runs 30

cargo run --release --no-default-features --features gpu --bin ppocr-gpu-bench -- \
  --kind det --size tiny --height 416 --width 736 \
  --warmup 5 --runs 30
```

Detector inputs use `[1, 3, 416, 736]`; recognizer inputs use
`[1, 3, 48, 320]`. Run benchmarks with `--release`, record the exact hardware,
operating system, Rust version, model size, thread count, warmup count, and
timed run count with any new result.

## Apple M4 GPU

Hardware: Apple M4 (10-core GPU, 16 GB unified memory), macOS 26.5, Rust
1.92.0. Five warmups and 30 timed runs per model.

| Model | p50 | p90 | Throughput at p50 |
| --- | ---: | ---: | ---: |
| medium detector | 173.197 ms | 174.520 ms | 5.77 frames/s |
| medium recognizer | 19.050 ms | 20.177 ms | 52.49 lines/s |
| small detector | 36.589 ms | 36.778 ms | 27.33 frames/s |
| small recognizer | 8.843 ms | 8.945 ms | 113.08 lines/s |
| tiny detector | 20.337 ms | 21.575 ms | 49.17 frames/s |
| tiny recognizer | 3.837 ms | 3.870 ms | 260.61 lines/s |

The synchronized validation forward produced finite, nonzero output with the
expected shape for every listed model. The GPU backend uses Metal on macOS and
Vulkan on other supported platforms.

## Apple M4 CPU

Hardware: Apple M4, macOS 26.5, Rust 1.92.0. One CPU worker, five warmups, and
30 timed runs per model.

| Model | p50 | p90 | Throughput at p50 |
| --- | ---: | ---: | ---: |
| medium detector | 178.041 ms | 178.850 ms | 5.62 frames/s |
| medium recognizer | 18.754 ms | 18.971 ms | 53.32 lines/s |
| small detector | 44.782 ms | 45.091 ms | 22.33 frames/s |
| small recognizer | 7.038 ms | 7.226 ms | 142.09 lines/s |
| tiny detector | 24.944 ms | 25.826 ms | 40.09 frames/s |
| tiny recognizer | 1.943 ms | 2.092 ms | 514.67 lines/s |

## Windows x86-64 CPU

Hardware: Intel Core i5-12600K, Windows build 26100, Rust 1.97.0
(`x86_64-pc-windows-msvc`). The single-worker run used five warmups and 30
timed runs.

| Model | Average latency | p95 | Throughput |
| --- | ---: | ---: | ---: |
| tiny detector | 54.386 ms | 58.855 ms | 18.39 frames/s |
| small detector | 121.699 ms | 133.223 ms | 8.22 frames/s |
| medium detector | 709.232 ms | 751.713 ms | 1.41 frames/s |
| tiny recognizer | 6.664 ms | 7.516 ms | 150.05 lines/s |
| small recognizer | 30.345 ms | 34.174 ms | 32.95 lines/s |
| medium recognizer | 121.375 ms | 132.975 ms | 8.24 lines/s |

The four-worker acceptance run used 20 warmups and 50 timed runs:

| Model | Average latency | p95 | Throughput |
| --- | ---: | ---: | ---: |
| tiny detector | 37.025 ms | 39.240 ms | 27.01 frames/s |
| small detector | 86.485 ms | 91.517 ms | 11.56 frames/s |
| medium detector | 508.864 ms | 532.813 ms | 1.97 frames/s |
| tiny recognizer | 4.683 ms | 5.395 ms | 213.56 lines/s |
| small recognizer | 24.176 ms | 27.648 ms | 41.36 lines/s |
| medium recognizer | 100.158 ms | 104.133 ms | 9.98 lines/s |

The repository configuration targets `x86-64-v3` for x86-64 builds, requiring
AVX2 and FMA. Do not compare these measurements with generic x86-64 binaries.
