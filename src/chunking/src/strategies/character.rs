//! `character` — tamanho fixo por caracteres (`CharacterTextSplitter`), separador `""` por
//! padrão (corte caractere a caractere, como o `ChunkCharacter` do chatbot).
//!
//! Opções: `separator` (`""`), `keep_separator` (`false`), `is_separator_regex` (`false`).

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::normalize_chunks;
use crate::core::splitter::{CharacterSplitter, KeepSeparator, TextSplitter};
use crate::error::Result;

pub struct ChunkCharacter {
    sizing: Sizing,
    splitter: CharacterSplitter,
}

impl ChunkCharacter {
    pub fn new(sizing: Sizing, separator: &str, keep: KeepSeparator, is_regex: bool) -> Result<Self> {
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_keep_separator(keep);
        Ok(Self { sizing, splitter: CharacterSplitter::new(base, separator, is_regex)? })
    }

    pub fn from_options(sizing: Sizing, mut o: Options) -> Result<Self> {
        let separator = o.string("separator")?.unwrap_or_default();
        let keep = o.keep_separator(KeepSeparator::No)?;
        let is_regex = o.bool("is_separator_regex", false)?;
        o.finish()?;
        Self::new(sizing, &separator, keep, is_regex)
    }
}

impl Chunker for ChunkCharacter {
    crate::sizing_accessors!("character");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if crate::core::pystr::strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks(self.splitter.split_text(text)?))
    }
}
