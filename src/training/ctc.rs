use anyhow::{Result, bail};
use candle_core::{IndexOp, Tensor};

fn logsumexp_small(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        bail!("logsumexp_small called with empty list");
    }
    if tensors.len() == 1 {
        return Ok(tensors[0].clone());
    }
    let stacked = Tensor::stack(tensors, 0)?;
    let max = stacked.max_all()?;
    let sum = stacked.broadcast_sub(&max)?.exp()?.sum_all()?;
    Ok((&sum.log()? + &max)?)
}

pub fn ctc_loss(
    log_probs: &Tensor,
    targets: &[Vec<usize>],
    input_lengths: &[usize],
    blank_id: usize,
) -> Result<Tensor> {
    let (batch, max_t, num_classes) = log_probs.dims3()?;
    if batch == 0 {
        bail!("empty batch");
    }
    if targets.len() != batch {
        bail!(
            "targets batch {} does not match logits batch {}",
            targets.len(),
            batch
        );
    }
    if input_lengths.len() != batch {
        bail!(
            "input lengths {} does not match logits batch {}",
            input_lengths.len(),
            batch
        );
    }

    let device = log_probs.device();
    let log_zero = Tensor::new(-1e9f32, device)?;

    let mut losses = Vec::with_capacity(batch);

    for b in 0..batch {
        let t_len = input_lengths[b];
        if t_len == 0 || t_len > max_t {
            bail!("invalid input length {t_len} for max_t {max_t}");
        }

        let target = &targets[b];
        let mut ext = Vec::with_capacity(target.len() * 2 + 1);
        ext.push(blank_id);
        for &label in target {
            if label >= num_classes {
                bail!("label id {label} out of range (classes {num_classes})");
            }
            ext.push(label);
            ext.push(blank_id);
        }

        let s_len = ext.len();
        let min_steps = min_steps_for_ctc(target);
        if t_len < min_steps {
            bail!(
                "input length {t_len} too short for target length {} (min steps {min_steps})",
                target.len()
            );
        }

        let mut prev = vec![log_zero.clone(); s_len];
        prev[0] = log_probs.i((b, 0, blank_id))?;
        if s_len > 1 {
            prev[1] = log_probs.i((b, 0, ext[1]))?;
        }

        for t in 1..t_len {
            let mut curr = vec![log_zero.clone(); s_len];
            for s in 0..s_len {
                let mut candidates = Vec::with_capacity(3);
                candidates.push(prev[s].clone());
                if s > 0 {
                    candidates.push(prev[s - 1].clone());
                }
                if s > 1 && ext[s] != blank_id && ext[s] != ext[s - 2] {
                    candidates.push(prev[s - 2].clone());
                }
                let sum = logsumexp_small(&candidates)?;
                let emit = log_probs.i((b, t, ext[s]))?;
                curr[s] = (&sum + &emit)?;
            }
            prev = curr;
        }

        let last = s_len - 1;
        let mut candidates = Vec::with_capacity(2);
        candidates.push(prev[last].clone());
        if last > 0 {
            candidates.push(prev[last - 1].clone());
        }
        let loglik = logsumexp_small(&candidates)?;
        let loss = (&loglik * -1f64)?;
        losses.push(loss);
    }

    let mut total = losses[0].clone();
    for loss in losses.iter().skip(1) {
        total = (&total + loss)?;
    }
    let mean = (&total / batch as f64)?;
    Ok(mean)
}

fn min_steps_for_ctc(target: &[usize]) -> usize {
    if target.is_empty() {
        return 0;
    }
    let mut repeats = 0usize;
    for i in 1..target.len() {
        if target[i] == target[i - 1] {
            repeats += 1;
        }
    }
    target.len() + repeats
}
