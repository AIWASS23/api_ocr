//! `markdown` — `MarkdownTextSplitter`: corte recursivo com os separadores de Markdown
//! (títulos, cercas de código, réguas horizontais, parágrafos), separador mantido no início.
//! Como no LangChain, esses separadores são usados como texto literal (não regex).
//!
//! Opções: `keep_separator` (`true`), `is_separator_regex` (`false`), `strip_whitespace`.

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, TextSplitter};
use crate::error::Result;

/// Splitter recursivo com os separadores de uma linguagem, mas literais (o comportamento de
/// `MarkdownTextSplitter`/`LatexTextSplitter`, que não ligam `is_separator_regex`).
pub(crate) fn language_literal(sizing: Sizing, language: &str, mut o: Options) -> Result<RecursiveSplitter> {
    let keep = o.keep_separator(KeepSeparator::Start)?;
    let is_regex = o.bool("is_separator_regex", false)?;
    let strip_ws = o.bool("strip_whitespace", true)?;
    o.ignore(&["add_start_index"]);
    o.finish()?;
    let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_keep_separator(keep).with_strip_whitespace(strip_ws);
    RecursiveSplitter::new(base, &crate::core::languages::separators_for(language)?, is_regex)
}

pub struct ChunkMarkdown {
    sizing: Sizing,
    splitter: RecursiveSplitter,
}

impl ChunkMarkdown {
    pub fn new(sizing: Sizing, o: Options) -> Result<Self> {
        Ok(Self { sizing, splitter: language_literal(sizing, "markdown", o)? })
    }
}

impl Chunker for ChunkMarkdown {
    crate::sizing_accessors!("markdown");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks(self.splitter.split_text(text)?))
    }
}
