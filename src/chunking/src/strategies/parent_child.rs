//! `parent_child` (nova) — *small-to-big*: o texto é dividido em pais grandes
//! (`parent_chunk_size`, por padrão 4× o `chunk_size`; por seções de Markdown quando há títulos)
//! e cada pai em filhos pequenos (`chunk_size`). Os FILHOS são indexados (busca precisa) e cada
//! um aponta para o pai (`parent_id`), que é o que vai para o LLM (contexto completo).
//!
//! [`ChunkParentChild::split`] devolve pais e filhos; `chunk`/`chunk_documents` devolvem os
//! filhos. Sem um armazenamento de pais, use `include_parent_content: true` para levar o texto
//! do pai no metadado de cada filho.
//!
//! Opções: `parent_chunk_size`, `parent_chunk_overlap` (0), `parent_strategy`
//! (`"auto"` | `"markdown_header"` | `"recursive"`), `include_parent_content` (`false`).
//! Metadados dos filhos: `parent_id`, `parent_index`, `child_index` (e `parent_content`).

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::strip;
use crate::document::Document;
use crate::error::{ChunkError, Result};
use crate::strategies::markdown_header::{MarkdownHeaderSplitter, default_markdown_headers};
use crate::strategies::recursive::{ChunkRecursive, enforce_size};

#[derive(Debug, Clone, Copy, PartialEq)]
enum ParentStrategy {
    Auto,
    MarkdownHeader,
    Recursive,
}

pub struct ChunkParentChild {
    sizing: Sizing,
    parent_sizing: Sizing,
    parent_strategy: ParentStrategy,
    include_parent_content: bool,
    child: ChunkRecursive,
}

/// Pais e filhos de um texto.
#[derive(Debug, Clone)]
pub struct ParentChildSplit {
    pub parents: Vec<Document>,
    pub children: Vec<Document>,
}

/// Id estável do pai (FNV-1a de 64 bits sobre posição + texto).
fn parent_id(index: usize, text: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in index.to_le_bytes().iter().chain(text.as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("p{index}-{h:016x}")
}

impl ChunkParentChild {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let parent_size = o.usize("parent_chunk_size")?.unwrap_or(sizing.chunk_size * 4);
        let parent_overlap = o.usize("parent_chunk_overlap")?.unwrap_or(0);
        let parent_strategy = match o.string("parent_strategy")?.as_deref() {
            None | Some("auto") => ParentStrategy::Auto,
            Some("markdown_header") => ParentStrategy::MarkdownHeader,
            Some("recursive") => ParentStrategy::Recursive,
            Some(other) => return Err(ChunkError::config(format!("parent_child: parent_strategy '{other}'"))),
        };
        let include_parent_content = o.bool("include_parent_content", false)?;
        o.finish()?;
        if parent_size <= sizing.chunk_size {
            return Err(ChunkError::config("parent_child: parent_chunk_size precisa ser maior que chunk_size"));
        }
        Ok(Self {
            sizing,
            parent_sizing: Sizing::new(parent_size, parent_overlap)?,
            parent_strategy,
            include_parent_content,
            child: ChunkRecursive::new(sizing)?,
        })
    }

    fn parents_of(&self, text: &str) -> Result<Vec<Document>> {
        let has_headers = text.split('\n').any(|l| {
            let t = l.trim_start();
            let h = t.chars().take_while(|c| *c == '#').count();
            (1..=6).contains(&h) && t[h..].starts_with(' ')
        });
        let by_headers = match self.parent_strategy {
            ParentStrategy::MarkdownHeader => true,
            ParentStrategy::Recursive => false,
            ParentStrategy::Auto => has_headers,
        };
        let mut parents = Vec::new();
        if by_headers {
            let splitter = MarkdownHeaderSplitter::new(default_markdown_headers(), false, false, Default::default())?;
            for s in splitter.split(text) {
                for piece in enforce_size(std::slice::from_ref(&s.content), self.parent_sizing)? {
                    parents.push(Document::with_metadata(piece, s.metadata()));
                }
            }
        } else {
            let recursive = ChunkRecursive::new(self.parent_sizing)?;
            parents.extend(recursive.chunk(text)?.into_iter().map(Document::new));
        }
        Ok(parents)
    }

    pub fn split(&self, text: &str) -> Result<ParentChildSplit> {
        let mut parents = Vec::new();
        let mut children = Vec::new();
        if strip(text).is_empty() {
            return Ok(ParentChildSplit { parents, children });
        }
        for (pi, mut parent) in self.parents_of(text)?.into_iter().enumerate() {
            let id = parent_id(pi, &parent.page_content);
            for (ci, child) in self.child.chunk(&parent.page_content)?.into_iter().enumerate() {
                let mut doc = Document::with_metadata(child, parent.metadata.clone())
                    .insert("parent_id", id.as_str())
                    .insert("parent_index", pi as u64)
                    .insert("child_index", ci as u64);
                if self.include_parent_content {
                    doc = doc.insert("parent_content", parent.page_content.as_str());
                }
                children.push(doc);
            }
            parent.metadata.insert("parent_id".into(), id.into());
            parent.metadata.insert("parent_index".into(), (pi as u64).into());
            parents.push(parent);
        }
        Ok(ParentChildSplit { parents, children })
    }
}

impl Chunker for ChunkParentChild {
    crate::sizing_accessors!("parent_child");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.split(text)?.children.into_iter().map(|d| d.page_content).collect())
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        Ok(self.split(text)?.children)
    }
}
