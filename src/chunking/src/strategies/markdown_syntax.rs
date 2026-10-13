//! `markdown_syntax` — port do `ExperimentalMarkdownSyntaxTextSplitter`: respeita a sintaxe do
//! Markdown, separando títulos, blocos de código cercados (cada um vira um chunk, com a
//! linguagem em `Code`) e réguas horizontais; depois impõe o tamanho.
//!
//! Como no LangChain, um bloco de código sem a cerca de fechamento é descartado.
//!
//! Opções: `headers_to_split_on` (padrão: `#`..`######` → `Header 1`..`Header 6`),
//! `strip_headers` (`false`), `return_each_line` (`false`).

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::Document;
use crate::error::Result;
use crate::core::section::Section;
use crate::strategies::recursive::enforce_size;

pub struct ChunkMarkdownSyntax {
    sizing: Sizing,
    splittable: Vec<(String, String)>,
    strip_headers: bool,
    return_each_line: bool,
}

/// `^(#{1,6}) (.*)` → (profundidade, texto).
fn match_header(line: &str) -> Option<(usize, String)> {
    let hashes = line.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&hashes) || !line[hashes..].starts_with(' ') {
        return None;
    }
    let rest = &line[hashes + 1..];
    // `.` não casa com `\n`.
    let text = rest.split('\n').next().unwrap_or("");
    Some((hashes, text.to_string()))
}

/// `^```(.*)` ou `^~~~(.*)`.
fn match_code(line: &str) -> Option<String> {
    let rest = line.strip_prefix("```").or_else(|| line.strip_prefix("~~~"))?;
    Some(rest.split('\n').next().unwrap_or("").to_string())
}

/// `^\*\*\*+\n`, `^---+\n`, `^___+\n`.
fn match_horz(line: &str) -> bool {
    let Some(body) = line.strip_suffix('\n') else { return false };
    ['*', '-', '_'].iter().any(|c| body.len() >= 3 && body.chars().all(|x| x == *c))
}

impl ChunkMarkdownSyntax {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let splittable = o.pairs("headers_to_split_on")?.filter(|h| !h.is_empty()).unwrap_or_else(|| {
            (1..=6).map(|l| ("#".repeat(l), format!("Header {l}"))).collect()
        });
        let strip_headers = o.bool("strip_headers", false)?;
        let return_each_line = o.bool("return_each_line", false)?;
        o.finish()?;
        Ok(Self { sizing, splittable, strip_headers, return_each_line })
    }

    fn key_for(&self, depth: usize) -> Option<&str> {
        let hashes = "#".repeat(depth);
        self.splittable.iter().find(|(h, _)| *h == hashes).map(|(_, n)| n.as_str())
    }

    pub fn split(&self, text: &str) -> Vec<Section> {
        let mut chunks: Vec<Section> = Vec::new();
        let mut current = Section { content: String::new(), headers: Vec::new() };
        let mut stack: Vec<(usize, String)> = Vec::new();
        let complete = |current: &mut Section, stack: &[(usize, String)], chunks: &mut Vec<Section>| {
            let c = std::mem::replace(current, Section { content: String::new(), headers: Vec::new() });
            if !c.content.is_empty() && !pystr::is_all_space(&c.content) {
                let mut headers = c.headers;
                for (depth, value) in stack {
                    if let Some(k) = self.key_for(*depth) {
                        match headers.iter_mut().find(|(hk, _)| hk == k) {
                            Some(slot) => slot.1 = value.clone(),
                            None => headers.push((k.to_string(), value.clone())),
                        }
                    }
                }
                chunks.push(Section { content: c.content, headers });
            }
        };
        let lines = pystr::splitlines(text, true);
        let mut i = 0;
        while i < lines.len() {
            let raw = lines[i];
            i += 1;
            let header = match_header(raw).filter(|(d, _)| self.key_for(*d).is_some());
            if let Some((depth, htext)) = header {
                complete(&mut current, &stack, &mut chunks);
                if !self.strip_headers {
                    current.content.push_str(raw);
                }
                if let Some(pos) = stack.iter().position(|(d, _)| *d >= depth) {
                    stack.truncate(pos);
                }
                stack.push((depth, htext));
            } else if let Some(lang) = match_code(raw) {
                complete(&mut current, &stack, &mut chunks);
                // `_resolve_code_chunk`: até a cerca de fechamento; sem ela, nada.
                let mut chunk = raw.to_string();
                let mut closed = false;
                while i < lines.len() {
                    let l = lines[i];
                    i += 1;
                    chunk.push_str(l);
                    if match_code(l).is_some() {
                        closed = true;
                        break;
                    }
                }
                current.content = if closed { chunk } else { String::new() };
                current.headers.push(("Code".into(), lang));
                complete(&mut current, &stack, &mut chunks);
            } else if match_horz(raw) {
                complete(&mut current, &stack, &mut chunks);
            } else {
                current.content.push_str(raw);
            }
        }
        complete(&mut current, &stack, &mut chunks);
        if self.return_each_line {
            return chunks
                .into_iter()
                .flat_map(|c| {
                    pystr::splitlines(&c.content, false)
                        .into_iter()
                        .filter(|l| !l.is_empty() && !pystr::is_all_space(l))
                        .map(|l| Section { content: l.to_string(), headers: c.headers.clone() })
                        .collect::<Vec<_>>()
                })
                .collect();
        }
        chunks
    }
}

impl Chunker for ChunkMarkdownSyntax {
    crate::sizing_accessors!("markdown_syntax");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let texts = pystr::normalize_chunks(self.split(text).into_iter().map(|s| s.content));
        enforce_size(&texts, self.sizing)
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        let mut out = Vec::new();
        for s in self.split(text) {
            for c in enforce_size(&pystr::normalize_chunks([s.content.as_str()]), self.sizing)? {
                out.push(Document::with_metadata(c, s.metadata()));
            }
        }
        Ok(out)
    }
}
