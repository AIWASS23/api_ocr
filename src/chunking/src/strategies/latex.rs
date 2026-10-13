//! `latex` — `LatexTextSplitter`: corte recursivo nos comandos de seção e ambientes do LaTeX
//! (`\chapter{`, `\section{`, `\begin{itemize}`...), separador mantido no início. Como no
//! LangChain, os separadores são texto literal.
//!
//! Opções: `keep_separator` (`true`), `is_separator_regex` (`false`), `strip_whitespace`.

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::RecursiveSplitter;
use crate::error::Result;
use crate::strategies::markdown::language_literal;

pub struct ChunkLatex {
    sizing: Sizing,
    splitter: RecursiveSplitter,
}

impl ChunkLatex {
    pub fn new(sizing: Sizing, o: Options) -> Result<Self> {
        Ok(Self { sizing, splitter: language_literal(sizing, "latex", o)? })
    }
}

impl Chunker for ChunkLatex {
    crate::sizing_accessors!("latex");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks(self.splitter.split_text(text)?))
    }
}
