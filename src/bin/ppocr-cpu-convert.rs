use anyhow::{Context, Result, bail};
use ppocr_rs::cpu::convert_onnx;
use std::{env, path::PathBuf};

fn main() -> Result<()> {
    let mut arguments = env::args_os().skip(1);
    let input = PathBuf::from(
        arguments
            .next()
            .context("usage: ppocr-cpu-convert INPUT.onnx OUTPUT.ppocr-cpu --shape N C H W")?,
    );
    let output = PathBuf::from(
        arguments
            .next()
            .context("usage: ppocr-cpu-convert INPUT.onnx OUTPUT.ppocr-cpu --shape N C H W")?,
    );
    ensure_flag(arguments.next(), "--shape")?;
    let mut shape = [0usize; 4];
    for dimension in &mut shape {
        *dimension = arguments
            .next()
            .context("--shape requires four positive dimensions")?
            .into_string()
            .map_err(|_| anyhow::anyhow!("shape dimension is not UTF-8"))?
            .parse()
            .context("parse shape dimension")?;
        if *dimension == 0 {
            bail!("shape dimensions must be positive");
        }
    }
    if arguments.next().is_some() {
        bail!("unexpected argument after shape");
    }
    convert_onnx(&input, &output, &shape)?;
    println!("wrote {} for input {shape:?}", output.display());
    Ok(())
}

fn ensure_flag(value: Option<std::ffi::OsString>, expected: &str) -> Result<()> {
    if value.as_deref() != Some(std::ffi::OsStr::new(expected)) {
        bail!("expected {expected}");
    }
    Ok(())
}
