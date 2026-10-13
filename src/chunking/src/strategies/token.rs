//! `token` — corte recursivo (`\n\n`, `\n`, `" "`, caractere) medindo em tokens de um
//! tokenizador Hugging Face (`RecursiveCharacterTextSplitter.from_huggingface_tokenizer`,
//! `keep_separator="end"`). Precisa de modelo: `manager.load_model("token")` ou injeção.
//!
//! Como no chatbot, `options` não é repassado a este splitter.

use std::sync::Arc;

use crate::chunker::{Chunker, Sizing};
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::{KeepSeparator, LengthFn, RecursiveSplitter, TextSplitter};
use crate::error::Result;
use crate::models::tokenizer::TextTokenizer;

pub struct ChunkToken {
    sizing: Sizing,
    splitter: RecursiveSplitter,
}

impl ChunkToken {
    pub fn new(tokenizer: Arc<dyn TextTokenizer>, sizing: Sizing) -> Result<Self> {
        let length: LengthFn = Arc::new(move |t: &str| tokenizer.count(t));
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?
            .with_length(length)
            .with_keep_separator(KeepSeparator::End);
        Ok(Self { sizing, splitter: RecursiveSplitter::new(base, &["\n\n", "\n", " ", ""], false)? })
    }
}

impl Chunker for ChunkToken {
    crate::sizing_accessors!("token");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks(self.splitter.split_text(text)?))
    }
}
