//! `html_header` — divide o HTML pelos títulos `h1`–`h6` (`HTMLHeaderTextSplitter`): cada bloco
//! leva o caminho de títulos ativos, que é prefixado ao texto (`"h1 > h2\ntexto"`, como o
//! `_document_texts(include_metadata=True)` do chatbot), e depois o tamanho é imposto.
//!
//! Opções: `headers_to_split_on` (`[["h1","h1"], ...]`), `return_each_element` (`false`).

use crate::chunker::{Chunker, Sizing};
use crate::core::html::{HElement, HNode, body_or_document};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::Document;
pub use crate::core::section::{Section, with_heading};
use crate::error::Result;
use crate::strategies::recursive::enforce_size;

pub struct ChunkHtmlHeader {
    sizing: Sizing,
    headers: Vec<(String, String)>,
    return_each_element: bool,
}

pub fn default_html_headers() -> Vec<(String, String)> {
    (1..=6).map(|l| (format!("h{l}"), format!("h{l}"))).collect()
}

/// Valor de `active_headers`: (texto, nível, profundidade no DOM).
struct Active {
    name: String,
    text: String,
    level: usize,
    depth: usize,
}

fn headers_of(active: &[Active]) -> Vec<(String, String)> {
    active.iter().map(|a| (a.name.clone(), a.text.clone())).collect()
}

impl ChunkHtmlHeader {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let mut headers = o.pairs("headers_to_split_on")?.unwrap_or_else(default_html_headers);
        let return_each_element = o.bool("return_each_element", false)?;
        o.finish()?;
        headers.sort_by_key(|(t, _)| t[1..].parse::<usize>().unwrap_or(9999));
        Ok(Self { sizing, headers, return_each_element })
    }

    /// `HTMLHeaderTextSplitter.split_text`.
    pub fn split_html(&self, html: &str) -> Vec<Section> {
        let doc = crate::core::html::parse(html);
        let body = body_or_document(html, &doc);
        let mut out = Vec::new();
        let mut active: Vec<Active> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let finalize = |current: &mut Vec<String>, active: &[Active]| -> Option<Section> {
            if current.is_empty() {
                return None;
            }
            let text = current.iter().filter(|l| !strip(l).is_empty()).cloned().collect::<Vec<_>>().join("  \n");
            current.clear();
            (!strip(&text).is_empty()).then(|| Section { content: text, headers: headers_of(active) })
        };
        // Pilha em profundidade, com a profundidade de cada nó (raiz `[document]` = 0).
        let base_depth = if std::ptr::eq(body, &doc) { 0 } else { depth_of(&doc, body).unwrap_or(0) };
        let mut stack: Vec<(&HElement, usize)> = vec![(body, base_depth)];
        while let Some((node, depth)) = stack.pop() {
            for c in node.children.iter().rev() {
                if let HNode::Element(e) = c {
                    stack.push((e, depth + 1));
                }
            }
            let texts: Vec<&str> = node.direct_strings().into_iter().map(pystr::strip).filter(|t| !t.is_empty()).collect();
            let node_text = texts.join(" ");
            if node_text.is_empty() {
                continue;
            }
            if let Some((_, name)) = self.headers.iter().find(|(t, _)| *t == node.name) {
                if !self.return_each_element
                    && let Some(d) = finalize(&mut current, &active)
                {
                    out.push(d);
                }
                let level = node.name[1..].parse::<usize>().unwrap_or(9999);
                active.retain(|a| a.level < level);
                active.retain(|a| a.name != *name);
                active.push(Active { name: name.clone(), text: node_text.clone(), level, depth });
                out.push(Section { content: node_text, headers: headers_of(&active) });
            } else {
                active.retain(|a| depth >= a.depth);
                if self.return_each_element {
                    out.push(Section { content: node_text, headers: headers_of(&active) });
                } else {
                    current.push(node_text);
                }
            }
        }
        if !self.return_each_element
            && let Some(d) = finalize(&mut current, &active)
        {
            out.push(d);
        }
        out
    }
}

/// Profundidade de `target` a partir de `root` (número de ancestrais).
pub(crate) fn depth_of(root: &HElement, target: &HElement) -> Option<usize> {
    fn go(node: &HElement, target: &HElement, d: usize) -> Option<usize> {
        if std::ptr::eq(node, target) {
            return Some(d);
        }
        node.child_elements().find_map(|c| go(c, target, d + 1))
    }
    go(root, target, 0)
}

impl Chunker for ChunkHtmlHeader {
    crate::sizing_accessors!("html_header");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = self.split_html(text).iter().map(with_heading).collect();
        enforce_size(&pystr::normalize_chunks(texts), self.sizing)
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        let mut out = Vec::new();
        for d in self.split_html(text) {
            for c in enforce_size(&pystr::normalize_chunks([with_heading(&d)]), self.sizing)? {
                out.push(Document::with_metadata(c, d.metadata()));
            }
        }
        Ok(out)
    }
}
