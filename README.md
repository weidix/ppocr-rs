# ppocr-rs

Native Rust inference for the released PP-OCRv6 Safetensors detector and
recognizer models. The primary `ppocr` command accepts a PNG or JPEG image and
returns detected text regions in reading order.

## Quick Start

Run OCR with the default tiny detector and recognizer:

```sh
cargo run --release --bin ppocr -- path/to/image.png
```

The first run downloads the pinned model packages into `models/`. Later runs
reuse the validated cache. Use `--format text` for newline-separated text
instead of JSON:

```sh
cargo run --release --bin ppocr -- path/to/image.jpg --format text
```

The JSON result has a stable image-level shape:

```json
{
  "source_size": [1920, 1080],
  "detector_input_size": [736, 416],
  "lines": [
    {
      "polygon": [[12.0, 24.0], [180.0, 24.0], [180.0, 58.0], [12.0, 58.0]],
      "detection_score": 0.98,
      "text": "example",
      "recognition_score": 0.97
    }
  ]
}
```

## Models

`models.json` is the single pinned catalog for all supported model packages.
Every download uses a fixed Hugging Face revision, writes through a temporary
file, and validates the size and SHA-256 digest of every required asset before
it becomes visible in the cache. A process lock prevents concurrent invocations
from corrupting the same cache.

The default model directory is `models`; override it with `--model-dir` or
`PPOCR_MODEL_DIR`:

```sh
PPOCR_MODEL_DIR="$HOME/.cache/ppocr-rs" \
  cargo run --release --bin ppocr -- path/to/image.png --model-size small
```

`--offline` fails instead of downloading missing models. `--verify-models`
recomputes all model digests before loading. The normal cache check verifies the
catalog completion marker and file sizes, so OCR startup does not repeatedly
read large weight files.

For custom converted or local checkpoints, supply explicit paths:

```sh
cargo run --release --bin ppocr -- path/to/image.png \
  --detector-model custom/det/model.safetensors \
  --recognizer-model custom/rec/model.safetensors \
  --dictionary custom/rec/inference.yml
```

When `--recognizer-model` is used without `--dictionary`, `ppocr` looks for an
`inference.yml` file beside the recognizer weights.

## Library API

The high-level CPU API keeps model resolution separate from inference:

```rust,no_run
use ppocr_rs::{ModelStore, OcrEngine, OcrOptions};

let store = ModelStore::new("models");
let engine = OcrEngine::load_from_store(&store, OcrOptions::default())?;
let result = engine.recognize_path("document.png")?;
println!("{}", result.text());
# Ok::<(), anyhow::Error>(())
```

`ppocr_rs::ocr` also exposes detector postprocessing, text rectification, crop
splitting, and CTC decoding for applications that need lower-level control.
`ppocr_rs::cpu` and `ppocr_rs::gpu` remain available as advanced direct-model
APIs.

## Backends

| Surface | CPU | GPU |
| --- | --- | --- |
| End-to-end `ppocr` image OCR | Yes | Not yet exposed as a high-level pipeline |
| Direct Safetensors model API | Yes | Yes |
| Benchmark command | `ppocr-cpu-bench` | `ppocr-gpu-bench` |

CPU inference is enabled by the default `cpu` feature. GPU inference is opt-in
with `--no-default-features --features gpu`; it uses Metal on macOS and Vulkan
on other supported platforms. The repository's x86-64 Cargo configuration
targets `x86-64-v3`, so AVX2 and FMA are required for those builds.

## Benchmarks

Both benchmark commands use the same pinned cache, `--kind det|rec`,
`--size medium|small|tiny`, `--model-dir`, `--offline`, and
`--verify-models` options. They accept explicit weights through `--model`
(CPU) or `--weights` (GPU).

```sh
cargo run --release --bin ppocr-cpu-bench -- \
  --kind det --size tiny --height 416 --width 736 --threads 4 --warmup 5 --runs 30

cargo run --release --no-default-features --features gpu --bin ppocr-gpu-bench -- \
  --kind det --size tiny --height 416 --width 736 --warmup 5 --runs 30
```

See [BENCHMARK.md](BENCHMARK.md) for recorded measurements and the complete
reproduction contract.

## Model Source

The pinned packages are published by
[PaddlePaddle on Hugging Face](https://huggingface.co/PaddlePaddle). Model
terms are defined by their upstream repositories; review those terms before
redistributing model files.
