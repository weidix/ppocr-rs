# ppocr-rs

PP-OCRv6 inference in Rust with two backends: native CPU kernels and WGPU.
Both backends load the released F32 Safetensors weights for the medium, small,
and tiny detector and recognizer models.

| Feature | Purpose | Default |
| --- | --- | --- |
| `cpu` | Native CPU inference | Yes |
| `cpu-profile` | Per-operation timings for the CPU backend | No |
| `gpu` | WGPU inference through Metal or Vulkan | No |

## CPU backend

Download the pinned model revisions, then run the benchmark:

```sh
./scripts/download-models.sh

cargo run --release --features cpu --bin ppocr-cpu-bench -- \
  --kind det --size tiny \
  --height 416 --width 736 --threads 4 --warmup 5 --runs 30
```

The benchmark defaults to
`models/<medium|small|tiny>-<det|rec>/model.safetensors`. Revisions, file sizes,
and SHA-256 values are recorded in `models.json`; use `--model` to override the
local model path.

The Rust API is available under `ppocr_rs::cpu`. On macOS, large pointwise
convolutions use Accelerate SGEMM. Windows MSVC builds use the `x86-64-v3`
AVX2/FMA baseline configured in `.cargo/config.toml`.

## GPU backend

The GPU backend selects Metal on macOS and Vulkan on other supported platforms:

```sh
cargo run --release --no-default-features --features gpu \
  --bin ppocr-gpu-bench -- \
  --model det --size tiny \
  --weights models/tiny-det/model.safetensors \
  --height 416 --width 736 --warmup 5 --runs 30
```

The Rust API is available under `ppocr_rs::gpu`.

See `BENCHMARK.md` for recorded CPU and GPU results and reproduction commands.
