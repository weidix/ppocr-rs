#![recursion_limit = "256"]

#[cfg(ppocr_burn_models)]
#[path = "../burn_runtime/benchmark.rs"]
mod benchmark;

#[cfg(ppocr_burn_models)]
fn main() -> anyhow::Result<()> {
    benchmark::run()
}

#[cfg(not(ppocr_burn_models))]
fn main() -> anyhow::Result<()> {
    if std::env::args_os()
        .skip(1)
        .any(|argument| matches!(argument.to_str(), Some("--help") | Some("-h")))
    {
        println!(
            "usage: ppocr-burn-bench --image PATH [--annotations JSONL] [--warmup N] [--runs N]"
        );
        return Ok(());
    }
    anyhow::bail!(
        "Burn benchmark models are not generated. Set PPOCR_BURN_DET_ONNX and \
         PPOCR_BURN_REC_ONNX to fixed-shape ONNX files, then rebuild with --features burn-bench."
    )
}
