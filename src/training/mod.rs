pub mod ctc;
pub mod data;
pub mod decode;
pub mod metrics;
pub mod model;
pub mod vocab;

#[cfg(all(feature = "candle-metal", feature = "candle-cuda"))]
compile_error!("Select only one backend feature: candle-metal or candle-cuda.");

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Var};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use candle_optimisers::Decay;
use candle_optimisers::adam::{Adam, ParamsAdam};
use clap::Parser;

use crate::training::ctc::ctc_loss;
use crate::training::data::{AugmentConfig, BatchStats, Dataset};
use crate::training::decode::{beam_search_decode, greedy_decode};
use crate::training::metrics::{char_error_rate, word_error_rate};
use crate::training::model::Crnn;
use crate::training::vocab::Vocab;

#[derive(Parser, Debug)]
#[command(
    name = "kuike-ocr-train",
    version,
    about = "Train a CRNN OCR model with Candle"
)]
struct Args {
    #[arg(long)]
    train_list: PathBuf,
    #[arg(long)]
    val_list: Option<PathBuf>,
    #[arg(long)]
    charset: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 32)]
    image_height: u32,
    #[arg(long, default_value_t = 0)]
    max_width: u32,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 8)]
    eval_batch_size: usize,
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    #[arg(long, default_value_t = 1e-5)]
    min_lr: f64,
    #[arg(long, default_value_t = 500)]
    warmup_steps: usize,
    #[arg(long, default_value = "cosine")]
    lr_schedule: String,
    #[arg(long, default_value_t = 5.0)]
    grad_clip: f64,
    #[arg(long, default_value_t = 0.0)]
    weight_decay: f64,
    #[arg(long)]
    decoupled_weight_decay: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "outputs")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 100)]
    log_every: usize,
    #[arg(long)]
    shuffle: bool,
    #[arg(long)]
    bucketed: bool,
    #[arg(long, default_value_t = 256)]
    bucket_size: usize,
    #[arg(long)]
    augment: bool,
    #[arg(long, default_value_t = 0.04)]
    aug_noise: f32,
    #[arg(long, default_value_t = 0.2)]
    aug_blur_prob: f32,
    #[arg(long, default_value_t = 0.1)]
    aug_brightness: f32,
    #[arg(long, default_value_t = 0.2)]
    aug_contrast: f32,
    #[arg(long, default_value_t = 0.15)]
    aug_erase_prob: f32,
    #[arg(long, default_value_t = 0.3)]
    aug_erase_max_fraction: f32,
    #[arg(long, default_value_t = 1)]
    beam_width: usize,
    #[arg(long, default_value_t = 25)]
    beam_top_k: usize,
}

struct EvalMetrics {
    loss: f64,
    cer: f64,
    wer: f64,
    exact_match: f64,
    sample: Option<String>,
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    let device = build_device(&args.device)?;

    let vocab = Vocab::from_file(&args.charset)
        .with_context(|| format!("failed to load charset from {}", args.charset.display()))?;

    let train = Dataset::from_tsv(&args.train_list).with_context(|| {
        format!(
            "failed to load train list from {}",
            args.train_list.display()
        )
    })?;
    let val = match &args.val_list {
        Some(path) => Some(
            Dataset::from_tsv(path)
                .with_context(|| format!("failed to load val list from {}", path.display()))?,
        ),
        None => None,
    };

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("failed to create output dir {}", args.output_dir.display()))?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Crnn::new(vb, vocab.size(), args.image_height as usize)?;

    let weight_decay = if args.weight_decay > 0.0 {
        Some(if args.decoupled_weight_decay {
            Decay::DecoupledWeightDecay(args.weight_decay)
        } else {
            Decay::WeightDecay(args.weight_decay)
        })
    } else {
        None
    };

    let vars = varmap.all_vars();
    let mut opt = Adam::new(
        vars.clone(),
        ParamsAdam {
            lr: args.lr,
            weight_decay,
            ..Default::default()
        },
    )?;

    let max_width = if args.max_width > 0 {
        Some(args.max_width)
    } else {
        None
    };

    let mut best_cer = f64::INFINITY;
    let mut global_step = 0usize;
    let base_seed = if args.seed == 0 {
        rand::random::<u64>()
    } else {
        args.seed
    };
    let total_steps =
        args.epochs * ((train.len().max(1) + args.batch_size - 1) / args.batch_size).max(1);

    let augment = AugmentConfig {
        enable: args.augment,
        noise: args.aug_noise,
        blur_prob: args.aug_blur_prob,
        max_brightness: args.aug_brightness,
        max_contrast: args.aug_contrast,
        erase_prob: args.aug_erase_prob,
        erase_max_fraction: args.aug_erase_max_fraction,
    };

    for epoch in 1..=args.epochs {
        let epoch_seed = base_seed.wrapping_add(epoch as u64);
        let (mut iter, stats) = train.batch_iter(
            args.batch_size,
            args.shuffle,
            device.clone(),
            args.image_height,
            max_width,
            -1.0,
            &vocab,
            args.bucketed,
            args.bucket_size,
            epoch_seed,
            augment.clone(),
        )?;
        log_dataset_stats("train", &stats);
        if stats.total == stats.skipped {
            anyhow::bail!("no valid training samples after filtering");
        }

        let mut step = 0usize;
        let mut total_loss = 0f64;
        let mut total_batches = 0usize;

        while let Some(batch) = iter.next() {
            let batch = batch?;
            let logits = model.forward(&batch.images)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, 2)?;
            let (_batch_size, time_steps, _) = log_probs.dims3()?;
            let input_lengths = batch.input_lengths.clone();
            if input_lengths.iter().any(|&len| len > time_steps) {
                anyhow::bail!("input length exceeds logits time steps");
            }

            let loss = ctc_loss(&log_probs, &batch.targets, &input_lengths, vocab.blank_id())
                .map_err(anyhow::Error::from)?;
            let loss_value = loss.to_vec0::<f32>()? as f64;

            let lr = scheduled_lr(
                args.lr,
                args.min_lr,
                args.warmup_steps,
                global_step,
                total_steps,
                &args.lr_schedule,
            );
            opt.set_learning_rate(lr);

            let mut grads = loss.backward().map_err(anyhow::Error::from)?;
            let grad_norm = if args.grad_clip > 0.0 {
                Some(clip_grad_norm(&mut grads, &vars, args.grad_clip)?)
            } else {
                None
            };
            opt.step(&grads).map_err(anyhow::Error::from)?;

            total_loss += loss_value;
            total_batches += 1;
            step += 1;
            global_step += 1;
            if step % args.log_every == 0 {
                let avg = total_loss / total_batches as f64;
                if let Some(grad_norm) = grad_norm {
                    println!(
                        "epoch {epoch} step {step} loss {avg:.4} lr {lr:.6} grad {grad_norm:.3}"
                    );
                } else {
                    println!("epoch {epoch} step {step} loss {avg:.4} lr {lr:.6}");
                }
            }
        }

        let avg_loss = if total_batches == 0 {
            0.0
        } else {
            total_loss / total_batches as f64
        };
        println!("epoch {epoch} train_loss {avg_loss:.4}");

        if let Some(val) = val.as_ref() {
            let metrics = evaluate(
                &model,
                val,
                &vocab,
                &device,
                args.image_height,
                max_width,
                args.eval_batch_size,
                args.beam_width,
                args.beam_top_k,
            )?;
            println!(
                "epoch {epoch} val_loss {loss:.4} cer {cer:.4} wer {wer:.4} acc {acc:.4}",
                loss = metrics.loss,
                cer = metrics.cer,
                wer = metrics.wer,
                acc = metrics.exact_match
            );
            if let Some(sample) = metrics.sample {
                println!("epoch {epoch} sample {sample}");
            }

            if metrics.cer < best_cer {
                best_cer = metrics.cer;
                let best_path = args.output_dir.join("best.safetensors");
                varmap.save(&best_path)?;
            }
        }

        let checkpoint = args.output_dir.join(format!("epoch-{epoch}.safetensors"));
        varmap.save(&checkpoint)?;
    }

    Ok(())
}

fn build_device(choice: &str) -> Result<Device> {
    match choice {
        "cpu" => Ok(Device::Cpu),
        "cuda" => build_cuda_device(),
        "metal" => build_metal_device(),
        other => anyhow::bail!("unknown device: {other}"),
    }
}

fn build_cuda_device() -> Result<Device> {
    #[cfg(feature = "candle-cuda")]
    {
        Device::new_cuda(0).context("cuda device not available")
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        anyhow::bail!("cuda backend not enabled; build with --features candle-cuda");
    }
}

fn build_metal_device() -> Result<Device> {
    #[cfg(feature = "candle-metal")]
    {
        Device::new_metal(0).context("metal device not available")
    }
    #[cfg(not(feature = "candle-metal"))]
    {
        anyhow::bail!("metal backend not enabled; build with --features candle-metal");
    }
}

fn scheduled_lr(
    base_lr: f64,
    min_lr: f64,
    warmup_steps: usize,
    step: usize,
    total_steps: usize,
    schedule: &str,
) -> f64 {
    if schedule == "none" {
        return base_lr;
    }
    let warmup_steps = warmup_steps.max(1);
    let step = step.min(total_steps.max(1));
    if step < warmup_steps {
        return base_lr * (step as f64 / warmup_steps as f64);
    }
    let progress = if total_steps > warmup_steps {
        (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64
    } else {
        1.0
    };
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    min_lr + (base_lr - min_lr) * cosine
}

fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f64> {
    let mut total = 0f64;
    for var in vars {
        if let Some(grad) = grads.get(var.as_tensor()) {
            let value = grad.sqr()?.sum_all()?.to_vec0::<f32>()? as f64;
            total += value;
        }
    }
    let norm = total.sqrt();
    if norm > max_norm {
        let scale = max_norm / (norm + 1e-6);
        for var in vars {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let scaled = (grad * scale)?;
                grads.insert(var.as_tensor(), scaled);
            }
        }
    }
    Ok(norm)
}

fn log_dataset_stats(name: &str, stats: &BatchStats) {
    if stats.skipped > 0 {
        println!(
            "{name} samples total {total} skipped {skipped}",
            total = stats.total,
            skipped = stats.skipped
        );
    }
}

fn evaluate(
    model: &Crnn,
    dataset: &Dataset,
    vocab: &Vocab,
    device: &Device,
    image_height: u32,
    max_width: Option<u32>,
    batch_size: usize,
    beam_width: usize,
    beam_top_k: usize,
) -> Result<EvalMetrics> {
    let mut total_loss = 0f64;
    let mut total_batches = 0usize;
    let mut total_char_edits = 0usize;
    let mut total_char_ref = 0usize;
    let mut total_word_edits = 0usize;
    let mut total_word_ref = 0usize;
    let mut exact_matches = 0usize;
    let mut total_samples = 0usize;
    let mut sample_out = None;

    let augment = AugmentConfig::default();
    let (mut iter, stats) = dataset.batch_iter(
        batch_size,
        false,
        device.clone(),
        image_height,
        max_width,
        -1.0,
        vocab,
        false,
        1,
        0,
        augment,
    )?;
    log_dataset_stats("val", &stats);

    while let Some(batch) = iter.next() {
        let batch = batch?;
        let logits = model.forward(&batch.images)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, 2)?;
        let (_batch_size, time_steps, _) = log_probs.dims3()?;
        let input_lengths = batch.input_lengths.clone();
        if input_lengths.iter().any(|&len| len > time_steps) {
            anyhow::bail!("input length exceeds logits time steps");
        }
        let loss = ctc_loss(&log_probs, &batch.targets, &input_lengths, vocab.blank_id())
            .map_err(anyhow::Error::from)?;
        let loss_value = loss.to_vec0::<f32>()? as f64;
        total_loss += loss_value;
        total_batches += 1;

        let decoded = if beam_width > 1 {
            beam_search_decode(&log_probs, vocab, beam_width, beam_top_k)?
        } else {
            greedy_decode(&log_probs, vocab)?
        };

        for (pred, target) in decoded.iter().zip(batch.texts.iter()) {
            let (edits, len) = char_error_rate(pred, target);
            total_char_edits += edits;
            total_char_ref += len;
            let (w_edits, w_len) = word_error_rate(pred, target);
            total_word_edits += w_edits;
            total_word_ref += w_len;
            if pred == target {
                exact_matches += 1;
            }
            total_samples += 1;
        }

        if sample_out.is_none() {
            if let (Some(pred), Some(target)) = (decoded.first(), batch.texts.first()) {
                sample_out = Some(format!("pred='{pred}' target='{target}'"));
            }
        }
    }

    let avg_loss = if total_batches == 0 {
        0.0
    } else {
        total_loss / total_batches as f64
    };
    let cer = if total_char_ref == 0 {
        0.0
    } else {
        total_char_edits as f64 / total_char_ref as f64
    };
    let wer = if total_word_ref == 0 {
        0.0
    } else {
        total_word_edits as f64 / total_word_ref as f64
    };
    let exact_match = if total_samples == 0 {
        0.0
    } else {
        exact_matches as f64 / total_samples as f64
    };

    Ok(EvalMetrics {
        loss: avg_loss,
        cer,
        wer,
        exact_match,
        sample: sample_out,
    })
}
