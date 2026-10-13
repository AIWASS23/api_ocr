//! `markdown_header` — port do `MarkdownHeaderTextSplitter`: divide pelos títulos `#`–`######`
//! (ignorando o que estiver dentro de cercas de código), agrupa as linhas de cada seção e
//! depois impõe o tamanho. Como no chatbot, os títulos ficam no texto (`strip_headers=false`).
//!
//! Opções: `headers_to_split_on` (`[["#","h1"], ...]`), `strip_headers` (`false`),
//! `return_each_line` (`false`), `custom_header_patterns` (`{"**": 1}`: linha `**texto**` vira
//! título de nível 1).

use indexmap::IndexMap;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pyre::{PyRegex, escape};
use crate::core::pystr::{self, strip};
use crate::document::Document;
use crate::error::Result;
use crate::core::section::Section;
use crate::strategies::recursive::enforce_size;

pub fn default_markdown_headers() -> Vec<(String, String)> {
    (1..=6).map(|l| ("#".repeat(l), format!("h{l}"))).collect()
}

/// O motor (`MarkdownHeaderTextSplitter`), compartilhado com `markdown_header_contextual`.
pub struct MarkdownHeaderSplitter {
    headers: Vec<(String, String)>,
    strip_headers: bool,
    return_each_line: bool,
    custom: IndexMap<String, usize>,
    custom_re: Vec<(String, PyRegex)>,
}

struct Line {
    content: String,
    meta: Vec<(String, String)>,
}

impl MarkdownHeaderSplitter {
    pub fn new(
        mut headers: Vec<(String, String)>,
        strip_headers: bool,
        return_each_line: bool,
        custom: IndexMap<String, usize>,
    ) -> Result<Self> {
        // Mais longos primeiro (`sorted(..., key=len, reverse=True)`, estável).
        headers.sort_by_key(|(s, _)| std::cmp::Reverse(s.chars().count()));
        let custom_re = custom
            .keys()
            .map(|sep| {
                let e = escape(sep);
                PyRegex::new(&format!("^{e}(?!{e})(.+?)(?<!{e}){e}$")).map(|r| (sep.clone(), r))
            })
            .collect::<Result<_>>()?;
        Ok(Self { headers, strip_headers, return_each_line, custom, custom_re })
    }

    fn is_custom(&self, line: &str, sep: &str) -> bool {
        let Some((_, re)) = self.custom_re.iter().find(|(s, _)| s == sep) else { return false };
        match re.match_start(line).ok().flatten() {
            Some(m) => {
                let (s, e) = m.groups[0].expect("grupo 1");
                let content = strip(&line[s..e]);
                !content.is_empty() && !content.replace(' ', "").chars().all(|c| sep.contains(c))
            }
            None => false,
        }
    }

    fn aggregate(&self, lines: Vec<Line>) -> Vec<Section> {
        let mut agg: Vec<Line> = Vec::new();
        for line in lines {
            match agg.last_mut() {
                Some(last) if last.meta == line.meta => {
                    last.content.push_str("  \n");
                    last.content.push_str(&line.content);
                }
                Some(last)
                    if last.meta.len() < line.meta.len()
                        && last.content.rsplit('\n').next().is_some_and(|l| l.starts_with('#'))
                        && !self.strip_headers =>
                {
                    last.content.push_str("  \n");
                    last.content.push_str(&line.content);
                    last.meta = line.meta;
                }
                _ => agg.push(line),
            }
        }
        agg.into_iter().map(|l| Section { content: l.content, headers: l.meta }).collect()
    }

    pub fn split(&self, text: &str) -> Vec<Section> {
        let mut out: Vec<Line> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut current_meta: Vec<(String, String)> = Vec::new();
        let mut stack: Vec<(usize, String, String)> = Vec::new(); // nível, nome, texto
        let mut initial: Vec<(String, String)> = Vec::new();
        let mut in_code = false;
        let mut fence = "";
        for raw in text.split('\n') {
            let stripped: String = strip(raw).chars().filter(|c| pystr::is_printable(*c)).collect();
            if !in_code {
                if stripped.starts_with("```") && stripped.matches("```").count() == 1 {
                    in_code = true;
                    fence = "```";
                } else if stripped.starts_with("~~~") {
                    in_code = true;
                    fence = "~~~";
                }
            } else if stripped.starts_with(fence) {
                in_code = false;
                fence = "";
            }
            if in_code {
                current.push(stripped);
                continue;
            }
            let mut matched = false;
            for (sep, name) in &self.headers {
                let standard = stripped.starts_with(sep.as_str())
                    && (stripped.len() == sep.len() || stripped[sep.len()..].starts_with(' '));
                let custom = self.is_custom(&stripped, sep);
                if standard || custom {
                    let level = self.custom.get(sep).copied().unwrap_or_else(|| sep.matches('#').count());
                    while stack.last().is_some_and(|(l, _, _)| *l >= level) {
                        let (_, popped, _) = stack.pop().expect("não vazio");
                        initial.retain(|(k, _)| *k != popped);
                    }
                    let header_text = if custom {
                        strip(&stripped[sep.len()..stripped.len() - sep.len()]).to_string()
                    } else {
                        strip(&stripped[sep.len()..]).to_string()
                    };
                    stack.push((level, name.clone(), header_text.clone()));
                    match initial.iter_mut().find(|(k, _)| k == name) {
                        Some(slot) => slot.1 = header_text,
                        None => initial.push((name.clone(), header_text)),
                    }
                    if !current.is_empty() {
                        out.push(Line { content: current.join("\n"), meta: current_meta.clone() });
                        current.clear();
                    }
                    if !self.strip_headers {
                        current.push(stripped.clone());
                    }
                    matched = true;
                    break;
                }
            }
            if !matched {
                if !stripped.is_empty() {
                    current.push(stripped);
                } else if !current.is_empty() {
                    out.push(Line { content: current.join("\n"), meta: current_meta.clone() });
                    current.clear();
                }
            }
            current_meta = initial.clone();
        }
        if !current.is_empty() {
            out.push(Line { content: current.join("\n"), meta: current_meta });
        }
        if self.return_each_line {
            out.into_iter().map(|l| Section { content: l.content, headers: l.meta }).collect()
        } else {
            self.aggregate(out)
        }
    }

    pub fn from_options(o: &mut Options, strip_default: bool) -> Result<Self> {
        let headers = o.pairs("headers_to_split_on")?.unwrap_or_else(default_markdown_headers);
        let strip_headers = o.bool("strip_headers", strip_default)?;
        let return_each_line = o.bool("return_each_line", false)?;
        let custom = o.take::<IndexMap<String, usize>>("custom_header_patterns")?.unwrap_or_default();
        Self::new(headers, strip_headers, return_each_line, custom)
    }
}

pub struct ChunkMarkdownHeader {
    sizing: Sizing,
    splitter: MarkdownHeaderSplitter,
}

impl ChunkMarkdownHeader {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let splitter = MarkdownHeaderSplitter::from_options(&mut o, false)?;
        o.finish()?;
        Ok(Self { sizing, splitter })
    }
}

impl Chunker for ChunkMarkdownHeader {
    crate::sizing_accessors!("markdown_header");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let texts = pystr::normalize_chunks(self.splitter.split(text).into_iter().map(|s| s.content));
        enforce_size(&texts, self.sizing)
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        let mut out = Vec::new();
        for s in self.splitter.split(text) {
            for c in enforce_size(&pystr::normalize_chunks([s.content.as_str()]), self.sizing)? {
                out.push(Document::with_metadata(c, s.metadata()));
            }
        }
        Ok(out)
    }
}
