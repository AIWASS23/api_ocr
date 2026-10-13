//! `ChunkerAdapter` — completa o contrato para um splitter externo dado como função
//! (o equivalente do adapter Python para callables e objetos com `split_text`).

use crate::chunker::{Chunker, Sizing};
use crate::core::pystr::normalize_chunks;
use crate::error::Result;

type SplitFn = dyn Fn(&str) -> Result<Vec<String>> + Send + Sync;

pub struct ChunkerAdapter {
    name: String,
    sizing: Sizing,
    split: Box<SplitFn>,
}

impl ChunkerAdapter {
    pub fn new(name: &str, sizing: Sizing, split: impl Fn(&str) -> Result<Vec<String>> + Send + Sync + 'static) -> Self {
        Self { name: name.to_string(), sizing, split: Box::new(split) }
    }
}

impl Chunker for ChunkerAdapter {
    fn name(&self) -> &str {
        &self.name
    }
    fn chunk_size(&self) -> usize {
        self.sizing.chunk_size
    }
    fn chunk_overlap(&self) -> usize {
        self.sizing.chunk_overlap
    }
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if crate::core::pystr::strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks((self.split)(text)?))
    }
}
