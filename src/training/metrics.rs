pub fn edit_distance<T: Eq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for (i, item_a) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, item_b) in b.iter().enumerate() {
            let cost = if item_a == item_b { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        prev.clone_from_slice(&curr);
    }

    prev[b.len()]
}

pub fn char_error_rate(pred: &str, target: &str) -> (usize, usize) {
    let pred_chars: Vec<char> = pred.chars().collect();
    let target_chars: Vec<char> = target.chars().collect();
    let edits = edit_distance(&pred_chars, &target_chars);
    (edits, target_chars.len().max(1))
}

pub fn word_error_rate(pred: &str, target: &str) -> (usize, usize) {
    let pred_words: Vec<&str> = pred.split_whitespace().collect();
    let target_words: Vec<&str> = target.split_whitespace().collect();
    let edits = edit_distance(&pred_words, &target_words);
    (edits, target_words.len().max(1))
}
