//! Base comum de `word`, `sentence`, `paragraph` e `line` (`_NaturalChunker` do chatbot):
//! `RecursiveCharacterTextSplitter` com separadores regex próprios e `keep_separator="end"`.
//! Não é uma estratégia; cada estratégia tem o seu arquivo.

use crate::chunker::Sizing;
use crate::core::options::Options;
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, TextSplitter};
use crate::error::Result;

pub struct NaturalSplitter {
    splitter: RecursiveSplitter,
}

impl NaturalSplitter {
    /// Opções: `separators` (substitui os padrões), `keep_separator` (`"end"`),
    /// `strip_whitespace` (`true`), `add_start_index` (ignorada).
    pub fn new(sizing: Sizing, default_separators: &[&str], mut o: Options) -> Result<Self> {
        let separators = o.strings("separators")?;
        let keep = o.keep_separator(KeepSeparator::End)?;
        let strip_ws = o.bool("strip_whitespace", true)?;
        o.ignore(&["add_start_index"]);
        o.finish()?;
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?
            .with_keep_separator(keep)
            .with_strip_whitespace(strip_ws);
        let splitter = match separators {
            Some(s) if !s.is_empty() => RecursiveSplitter::new(base, &s, true)?,
            _ => RecursiveSplitter::new(base, default_separators, true)?,
        };
        Ok(Self { splitter })
    }

    pub fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        Ok(normalize_chunks(self.splitter.split_text(text)?))
    }
}

/// Declara uma estratégia "natural" (arquivo próprio, separadores próprios).
#[macro_export]
macro_rules! natural_strategy {
    ($ty:ident, $name:literal, $seps:expr) => {
        pub struct $ty {
            sizing: $crate::chunker::Sizing,
            inner: $crate::strategies::natural::NaturalSplitter,
        }

        impl $ty {
            pub const SEPARATORS: &'static [&'static str] = $seps;

            pub fn new(sizing: $crate::chunker::Sizing) -> $crate::error::Result<Self> {
                Self::from_options(sizing, $crate::core::options::Options::new($name, &Default::default()))
            }

            pub fn from_options(
                sizing: $crate::chunker::Sizing,
                options: $crate::core::options::Options,
            ) -> $crate::error::Result<Self> {
                Ok(Self { sizing, inner: $crate::strategies::natural::NaturalSplitter::new(sizing, Self::SEPARATORS, options)? })
            }
        }

        impl $crate::chunker::Chunker for $ty {
            $crate::sizing_accessors!($name);

            fn chunk(&self, text: &str) -> $crate::error::Result<Vec<String>> {
                self.inner.chunk(text)
            }
        }
    };
}
