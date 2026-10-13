//! `html_section` — divide o HTML em seções pelos títulos (`HTMLSectionSplitter`). Antes, como
//! a XSLT do LangChain, todo elemento com `font-size` maior que 20px vira `<h1>`. O texto da
//! seção é o título seguido de tudo até o próximo título; o caminho de títulos é prefixado e o
//! tamanho é imposto.
//!
//! Diferenças intencionais em relação ao Python:
//! * o Python deixa o marcador literal `#TITLE#` no início da primeira seção (texto antes do
//!   primeiro título); aqui ele vira o `<title>` do documento (ou `title_placeholder`), ou some;
//! * o Python passa o HTML pela serialização do lxml, que insere `\n` entre elementos de bloco;
//!   aqui o texto sai sem esses espaços extras (só espaço em branco muda).
//!
//! Opções: `headers_to_split_on`, `title_placeholder` (título da primeira seção; padrão: o
//! `<title>` do documento). Opções do splitter recursivo interno do LangChain
//! (`separators`, `keep_separator`...) são aceitas e ignoradas, como no Python.

use crate::chunker::{Chunker, Sizing};
use crate::core::html::{HElement, HNode};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::Document;
use crate::error::{ChunkError, Result};
use crate::core::section::{Section, with_heading};
use crate::strategies::html_header::default_html_headers;
use crate::strategies::recursive::enforce_size;

pub struct ChunkHtmlSection {
    sizing: Sizing,
    headers: Vec<(String, String)>,
    title_placeholder: Option<String>,
}

/// `number()` do XPath 1.0: espaços nas pontas, sinal opcional, dígitos e ponto.
fn xpath_number(s: &str) -> f64 {
    let t = s.trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\r'));
    let body = t.strip_prefix('-').unwrap_or(t);
    let valid = !body.is_empty()
        && body.chars().all(|c| c.is_ascii_digit() || c == '.')
        && body.matches('.').count() <= 1
        && body != ".";
    if valid { t.parse::<f64>().unwrap_or(f64::NAN) } else { f64::NAN }
}

/// A XSLT `converting_to_header.xslt`: `font-size` > 20 (px) vira `<h1>`.
fn promote_large_text(node: &mut HElement) {
    if let Some(style) = node.attr("style")
        && style.contains("font-size")
    {
        let after = style.split_once("font-size:").map_or("", |(_, a)| a);
        let value = after.split_once("px").map_or("", |(v, _)| v);
        if xpath_number(value) > 20.0 {
            node.name = "h1".into();
        }
    }
    for c in node.children.iter_mut() {
        if let HNode::Element(e) = c {
            promote_large_text(e);
        }
    }
}

/// Sequência em ordem de documento: início de elemento ou texto.
enum Event<'a> {
    Start(&'a HElement),
    Str(&'a str),
}

fn flatten<'a>(node: &'a HElement, out: &mut Vec<Event<'a>>) {
    for c in &node.children {
        match c {
            HNode::Element(e) => {
                out.push(Event::Start(e));
                flatten(e, out);
            }
            HNode::Text(t) | HNode::Comment(t) => out.push(Event::Str(t)),
        }
    }
}

impl ChunkHtmlSection {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let headers = o.pairs("headers_to_split_on")?.unwrap_or_else(default_html_headers);
        let title_placeholder = o.string("title_placeholder")?;
        o.ignore(&["separators", "keep_separator", "is_separator_regex", "strip_whitespace", "add_start_index"]);
        o.finish()?;
        if !headers.iter().any(|(t, _)| t == "h1") {
            return Err(ChunkError::config("html_section: headers_to_split_on precisa incluir 'h1'"));
        }
        Ok(Self { sizing, headers, title_placeholder })
    }

    pub fn split_html(&self, html: &str) -> Vec<Section> {
        let mut doc = crate::core::html::parse(html);
        promote_large_text(&mut doc);
        let title = self.title_placeholder.clone().or_else(|| {
            doc.find("title").map(|t| pystr::strip(&t.text_content()).to_string()).filter(|t| !t.is_empty())
        });
        let mut events = Vec::new();
        flatten(&doc, &mut events);
        let is_split = |e: &HElement| e.name == "body" || self.headers.iter().any(|(t, _)| *t == e.name);
        let starts: Vec<usize> = events
            .iter()
            .enumerate()
            .filter_map(|(i, ev)| matches!(ev, Event::Start(e) if is_split(e)).then_some(i))
            .collect();
        let mut out = Vec::new();
        for (k, &pos) in starts.iter().enumerate() {
            let Event::Start(el) = events[pos] else { continue };
            let end = starts.get(k + 1).copied().unwrap_or(events.len());
            let strings: Vec<&str> = events[pos + 1..end]
                .iter()
                .filter_map(|ev| match ev {
                    Event::Str(s) => Some(*s),
                    Event::Start(_) => None,
                })
                .collect();
            let content = strip(&strings.join(" ")).to_string();
            if content.is_empty() {
                continue;
            }
            let (header, tag) = if k == 0 {
                (title.clone().unwrap_or_default(), "h1".to_string())
            } else {
                (strip(&el.text_content()).to_string(), el.name.clone())
            };
            let name = self.headers.iter().find(|(t, _)| *t == tag).map(|(_, n)| n.clone()).unwrap_or(tag);
            out.push(Section { content, headers: vec![(name, header)] });
        }
        out
    }
}

impl Chunker for ChunkHtmlSection {
    crate::sizing_accessors!("html_section");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = self.split_html(text).iter().map(with_heading).collect();
        enforce_size(&pystr::normalize_chunks(texts), self.sizing)
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        let mut out = Vec::new();
        for s in self.split_html(text) {
            for c in enforce_size(&pystr::normalize_chunks([with_heading(&s)]), self.sizing)? {
                out.push(Document::with_metadata(c, s.metadata()));
            }
        }
        Ok(out)
    }
}
