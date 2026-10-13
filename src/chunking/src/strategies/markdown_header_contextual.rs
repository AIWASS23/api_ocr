//! `markdown_header_contextual` (nova) — divide por títulos de Markdown como
//! `markdown_header`, mas repete o caminho de títulos (`Doc > Capítulo > Seção`) no início de
//! CADA pedaço, inclusive quando uma seção longa é subdividida. No `markdown_header` original,
//! só o primeiro pedaço da seção tem o título; os demais perdem o contexto.
//!
//! O caminho conta no `chunk_size`: o corpo de cada seção é cortado com o espaço que sobra. Se o
//! caminho passar da metade do `chunk_size`, os títulos mais externos são omitidos (`… > `).
//!
//! Opções: `headers_to_split_on`, `title` (raiz do caminho, ex.: nome do documento),
//! `separator` (`" > "`), `strip_headers` (`true`: o título já está no caminho),
//! `custom_header_patterns`. Metadados: os títulos (`h1`, `h2`...) e `breadcrumb`.

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::Document;
use crate::error::Result;
use crate::strategies::markdown_header::MarkdownHeaderSplitter;
use crate::strategies::recursive::enforce_size;

pub struct ChunkMarkdownHeaderContextual {
    sizing: Sizing,
    splitter: MarkdownHeaderSplitter,
    title: Option<String>,
    separator: String,
}

impl ChunkMarkdownHeaderContextual {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let splitter = MarkdownHeaderSplitter::from_options(&mut o, true)?;
        let title = o.string("title")?.filter(|t| !strip(t).is_empty());
        let separator = o.string("separator")?.unwrap_or_else(|| " > ".into());
        o.finish()?;
        Ok(Self { sizing, splitter, title, separator })
    }

    /// Caminho que cabe em metade do `chunk_size`, omitindo os níveis mais externos.
    fn breadcrumb(&self, crumbs: &[String]) -> String {
        let limit = self.sizing.chunk_size / 2;
        let full = crumbs.join(&self.separator);
        if pystr::len(&full) <= limit {
            return full;
        }
        for skip in 1..crumbs.len() {
            let candidate = format!("…{}{}", self.separator, crumbs[skip..].join(&self.separator));
            if pystr::len(&candidate) <= limit {
                return candidate;
            }
        }
        // Nem o último título cabe: corta o texto dele.
        let last = crumbs.last().cloned().unwrap_or_default();
        last.chars().take(limit.saturating_sub(1)).collect::<String>() + "…"
    }
}

impl Chunker for ChunkMarkdownHeaderContextual {
    crate::sizing_accessors!("markdown_header_contextual");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.chunk_documents(text)?.into_iter().map(|d| d.page_content).collect())
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for section in self.splitter.split(text) {
            let body = strip(&section.content);
            if body.is_empty() {
                continue;
            }
            let crumbs: Vec<String> = self
                .title
                .iter()
                .cloned()
                .chain(section.headers.iter().map(|(_, v)| v.clone()))
                .filter(|c| !strip(c).is_empty())
                .collect();
            let prefix = if crumbs.is_empty() { String::new() } else { self.breadcrumb(&crumbs) };
            let reserved = if prefix.is_empty() { 0 } else { pystr::len(&prefix) + 2 };
            let budget = self.sizing.chunk_size.saturating_sub(reserved).max(1);
            let body_sizing = Sizing::new(budget, self.sizing.chunk_overlap.min(budget - 1))?;
            for piece in enforce_size(&[body.to_string()], body_sizing)? {
                let content = if prefix.is_empty() { piece } else { format!("{prefix}\n\n{piece}") };
                let mut meta = section.metadata();
                meta.insert("breadcrumb".into(), prefix.clone().into());
                out.push(Document::with_metadata(content, meta));
            }
        }
        Ok(out)
    }
}
