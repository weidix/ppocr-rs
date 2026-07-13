#![recursion_limit = "256"]

#[cfg(ppocr_burn_models)]
#[path = "../burn_runtime/inference.rs"]
mod inference;

#[cfg(ppocr_burn_models)]
fn main() -> anyhow::Result<()> {
    inference::run()
}

#[cfg(not(ppocr_burn_models))]
fn main() -> anyhow::Result<()> {
    if std::env::args_os()
        .skip(1)
        .any(|argument| matches!(argument.to_str(), Some("--help") | Some("-h")))
    {
        println!(
            "usage: ppocr-burn --image PATH --dict PATH [--output PATH] [--det-threshold F32] [--box-threshold F32] [--unclip-ratio F32] [--min-area N] [--max-boxes N]"
        );
        return Ok(());
    }
    anyhow::bail!(
        "Burn OCR models are not generated. Set PPOCR_BURN_DET_ONNX and PPOCR_BURN_REC_ONNX to fixed-shape ONNX files, then rebuild with --features burn-infer."
    )
}
