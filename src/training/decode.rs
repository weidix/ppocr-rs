use std::collections::HashMap;

use anyhow::Result;
use candle_core::Tensor;

use crate::training::vocab::Vocab;

const NEG_INFINITY: f32 = -1e9;

pub fn greedy_decode(log_probs: &Tensor, vocab: &Vocab) -> Result<Vec<String>> {
    let data = log_probs.to_vec3::<f32>()?;
    let mut outputs = Vec::with_capacity(data.len());
    let blank_id = vocab.blank_id();

    for seq in data {
        let mut out = String::new();
        let mut prev = None;
        for step in seq {
            let mut best_id = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &val) in step.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_id = idx;
                }
            }
            if best_id != blank_id && prev != Some(best_id) {
                out.push_str(vocab.token(best_id));
            }
            prev = Some(best_id);
        }
        outputs.push(out);
    }

    Ok(outputs)
}

pub fn beam_search_decode(
    log_probs: &Tensor,
    vocab: &Vocab,
    beam_width: usize,
    top_k: usize,
) -> Result<Vec<String>> {
    let data = log_probs.to_vec3::<f32>()?;
    let blank_id = vocab.blank_id();
    let mut outputs = Vec::with_capacity(data.len());

    for seq in data {
        let mut beam: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
        beam.insert(Vec::new(), (0.0, NEG_INFINITY));

        for step in seq {
            let mut top_indices: Vec<usize> = (0..step.len()).collect();
            top_indices.sort_by(|&a, &b| step[b].partial_cmp(&step[a]).unwrap());
            top_indices.truncate(top_k.max(1));

            let mut new_beam: HashMap<Vec<usize>, (f32, f32)> = HashMap::new();
            for (prefix, (pb, pnb)) in &beam {
                let total = logsumexp(*pb, *pnb);
                for &c in &top_indices {
                    let logp = step[c];
                    if c == blank_id {
                        let entry = new_beam
                            .entry(prefix.clone())
                            .or_insert((NEG_INFINITY, NEG_INFINITY));
                        entry.0 = logsumexp(entry.0, total + logp);
                        continue;
                    }

                    let last = prefix.last().copied();
                    if last == Some(c) {
                        let entry = new_beam
                            .entry(prefix.clone())
                            .or_insert((NEG_INFINITY, NEG_INFINITY));
                        entry.1 = logsumexp(entry.1, pb + logp);

                        let mut extended = prefix.clone();
                        extended.push(c);
                        let entry = new_beam
                            .entry(extended)
                            .or_insert((NEG_INFINITY, NEG_INFINITY));
                        entry.1 = logsumexp(entry.1, pnb + logp);
                    } else {
                        let mut extended = prefix.clone();
                        extended.push(c);
                        let entry = new_beam
                            .entry(extended)
                            .or_insert((NEG_INFINITY, NEG_INFINITY));
                        entry.1 = logsumexp(entry.1, total + logp);
                    }
                }
            }

            let mut entries: Vec<(Vec<usize>, (f32, f32))> = new_beam.into_iter().collect();
            entries.sort_by(|a, b| {
                let a_score = logsumexp(a.1.0, a.1.1);
                let b_score = logsumexp(b.1.0, b.1.1);
                b_score.partial_cmp(&a_score).unwrap()
            });
            entries.truncate(beam_width.max(1));

            beam = entries.into_iter().collect();
        }

        let best = beam
            .iter()
            .max_by(|a, b| {
                let a_score = logsumexp(a.1.0, a.1.1);
                let b_score = logsumexp(b.1.0, b.1.1);
                a_score.partial_cmp(&b_score).unwrap()
            })
            .map(|(seq, _)| seq.clone())
            .unwrap_or_default();

        outputs.push(vocab.decode(&best));
    }

    Ok(outputs)
}

fn logsumexp(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    if m <= NEG_INFINITY / 2.0 {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}
