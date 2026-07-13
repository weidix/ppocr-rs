use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result, bail};

const BLANK_TOKEN: &str = "<blank>";

#[derive(Debug, Clone)]
pub struct Vocab {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, usize>,
    blank_id: usize,
}

impl Vocab {
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open charset {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();

        id_to_token.push(BLANK_TOKEN.to_string());
        token_to_id.insert(BLANK_TOKEN.to_string(), 0);

        for (line_no, line) in reader.lines().enumerate() {
            let line = line.with_context(|| format!("read line {line_no}"))?;
            let token = line.trim();
            if token.is_empty() {
                continue;
            }
            if token_to_id.contains_key(token) {
                bail!("duplicate token '{token}' in charset");
            }
            let id = id_to_token.len();
            id_to_token.push(token.to_string());
            token_to_id.insert(token.to_string(), id);
        }

        Ok(Self {
            id_to_token,
            token_to_id,
            blank_id: 0,
        })
    }

    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    pub fn token(&self, id: usize) -> &str {
        self.id_to_token.get(id).map(|s| s.as_str()).unwrap_or("")
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let mut ids = Vec::with_capacity(text.chars().count());
        for ch in text.chars() {
            let token = ch.to_string();
            let id = self
                .token_to_id
                .get(&token)
                .copied()
                .with_context(|| format!("token '{token}' not in charset"))?;
            ids.push(id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == self.blank_id {
                continue;
            }
            out.push_str(self.token(id));
        }
        out
    }
}
