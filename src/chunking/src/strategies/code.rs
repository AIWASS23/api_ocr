//! `code` — código-fonte pela linguagem (`RecursiveCharacterTextSplitter.from_language`,
//! separador no fim): corta em classes, funções, blocos e linhas. Linguagens: as do LangChain
//! (cpp, go, java, kotlin, js, ts, php, proto, python, r, rst, ruby, rust, scala, swift,
//! markdown, latex, html, sol, csharp, cobol, c, lua, haskell, elixir, powershell,
//! visualbasic6). `perl` está na lista do LangChain mas não tem separadores (erro, como lá).
//!
//! Opções: `language` (padrão: `language` da configuração ou `python`), `strip_whitespace`.

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, TextSplitter};
use crate::error::{ChunkError, Result};

pub struct ChunkCode {
    sizing: Sizing,
    name: &'static str,
    splitter: RecursiveSplitter,
}

impl ChunkCode {
    pub fn new(sizing: Sizing, language: &str, name: &'static str, mut o: Options) -> Result<Self> {
        let strip_ws = o.bool("strip_whitespace", true)?;
        o.ignore(&["add_start_index"]);
        o.finish()?;
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?
            .with_keep_separator(KeepSeparator::End)
            .with_strip_whitespace(strip_ws);
        let splitter = RecursiveSplitter::from_language(base, language)
            .map_err(|_| ChunkError::config(format!("Unsupported source-code language '{language}'")))?;
        Ok(Self { sizing, name, splitter })
    }
}

impl Chunker for ChunkCode {
    fn name(&self) -> &str {
        self.name
    }
    fn chunk_size(&self) -> usize {
        self.sizing.chunk_size
    }
    fn chunk_overlap(&self) -> usize {
        self.sizing.chunk_overlap
    }

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks(self.splitter.split_text(text)?))
    }
}
