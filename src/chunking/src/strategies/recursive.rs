//! `recursive` — corte recursivo por estrutura (`RecursiveCharacterTextSplitter`): tenta
//! `\n\n`, depois `\n`, `" "` e por fim caractere. Com `language` (ou `options.language`), usa
//! os separadores daquela linguagem de programação/marcação.
//!
//! Opções: `separators`, `keep_separator` (`false`), `is_separator_regex` (`false`),
//! `strip_whitespace` (`true`), `add_start_index` (aceita e ignora: só afeta metadados).

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, TextSplitter};
use crate::error::{ChunkError, Result};

pub struct ChunkRecursive {
    sizing: Sizing,
    splitter: RecursiveSplitter,
}

impl ChunkRecursive {
    /// Separadores padrão, sem manter separador (o `ChunkRecursiveCharacter` do chatbot).
    pub fn new(sizing: Sizing) -> Result<Self> {
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?;
        Ok(Self { sizing, splitter: RecursiveSplitter::new(base, &[] as &[&str], false)? })
    }

    pub fn from_options(sizing: Sizing, language: Option<&str>, mut o: Options) -> Result<Self> {
        let separators = o.strings("separators")?;
        let keep = o.keep_separator(KeepSeparator::No)?;
        let is_regex = o.bool("is_separator_regex", false)?;
        let strip_ws = o.bool("strip_whitespace", true)?;
        o.ignore(&["add_start_index"]);
        o.finish()?;
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?
            .with_keep_separator(keep)
            .with_strip_whitespace(strip_ws);
        let splitter = match (separators, language) {
            (Some(seps), _) => RecursiveSplitter::new(base, &seps, is_regex)?,
            (None, Some(lang)) => RecursiveSplitter::from_language(base, lang)
                .map_err(|_| ChunkError::config(format!("Unsupported recursive language '{lang}'")))?,
            (None, None) => RecursiveSplitter::new(base, &[] as &[&str], is_regex)?,
        };
        Ok(Self { sizing, splitter })
    }

    /// Corte sem a normalização final (usado por outras estratégias para impor tamanho).
    pub fn split_raw(&self, text: &str) -> Result<Vec<String>> {
        self.splitter.split_text(text)
    }
}

impl Chunker for ChunkRecursive {
    crate::sizing_accessors!("recursive");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks(self.splitter.split_text(text)?))
    }
}

/// `_enforce_size` do chatbot: corta cada chunk que passar do tamanho, mantendo o separador no
/// fim, e normaliza. Usado pelas estratégias estruturais (Markdown, HTML, JSON...).
pub fn enforce_size(chunks: &[String], sizing: Sizing) -> Result<Vec<String>> {
    let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_keep_separator(KeepSeparator::End);
    let splitter = RecursiveSplitter::new(base, &[] as &[&str], false)?;
    let mut out = Vec::new();
    for c in chunks {
        out.extend(splitter.split_text(c)?);
    }
    Ok(normalize_chunks(out))
}
