//! `html_semantic` — port do `HTMLSemanticPreservingSplitter`: divide por títulos mantendo
//! inteiros os elementos de `elements_to_preserve` (tabelas, listas, `pre`, `code`), converte
//! links em `[texto](href)` e imagens em `![image:src](src)`, e só corta o texto entre esses
//! elementos. Como no `ChunkHTMLSemantic` do chatbot, o título é prefixado e o tamanho imposto.
//!
//! Opções: `headers_to_split_on`, `elements_to_preserve` (`["table","ul","ol","pre","code"]`),
//! `preserve_links` (`true`), `preserve_images` (`true`), `preserve_videos`, `preserve_audio`,
//! `separators`, `keep_separator` (`true`), `normalize_text`, `stopword_removal`,
//! `stopword_lang` (`english`; lista de stopwords do NLTK), `external_metadata`,
//! `allowlist_tags`, `denylist_tags`, `preserve_parent_metadata`.

use std::collections::HashSet;

use crate::chunker::{Chunker, Sizing};
use crate::core::html::{HElement, HNode};
use crate::core::options::Options;
use crate::core::pyre::PyRegex;
use crate::core::pystr::{self, strip};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, TextSplitter};
use crate::document::Document;
use crate::error::{ChunkError, Result};
use crate::core::section::{Section, with_heading};
use crate::strategies::html_header::default_html_headers;
use crate::strategies::recursive::enforce_size;

pub struct ChunkHtmlSemantic {
    sizing: Sizing,
    headers: Vec<(String, String)>,
    preserve: Vec<String>,
    preserve_links: bool,
    preserve_images: bool,
    preserve_videos: bool,
    preserve_audio: bool,
    normalize_text: bool,
    stopwords: Option<HashSet<String>>,
    external: Vec<(String, String)>,
    allowlist: Option<Vec<String>>,
    denylist: Option<Vec<String>>,
    splitter: RecursiveSplitter,
    non_word: PyRegex,
    spaces: PyRegex,
}

/// Lista de stopwords do NLTK (`corpora/stopwords/<idioma>`).
fn load_stopwords(lang: &str) -> Result<HashSet<String>> {
    crate::models::punkt::nltk_data_paths(None)
        .into_iter()
        .map(|p| p.join("corpora/stopwords").join(lang))
        .find(|p| p.is_file())
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.lines().map(str::to_string).collect())
        .ok_or_else(|| ChunkError::Model(format!("Resource stopwords not found (corpora/stopwords/{lang})")))
}

struct State {
    sections: Vec<Section>,
    headers: Vec<(String, String)>,
    content: Vec<String>,
    preserved: Vec<(String, String)>,
    placeholders: usize,
}

impl ChunkHtmlSemantic {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let mut headers = o.pairs("headers_to_split_on")?.unwrap_or_else(default_html_headers);
        headers.sort();
        let preserve = o
            .strings("elements_to_preserve")?
            .unwrap_or_else(|| ["table", "ul", "ol", "pre", "code"].map(String::from).to_vec());
        let preserve_links = o.bool("preserve_links", true)?;
        let preserve_images = o.bool("preserve_images", true)?;
        let preserve_videos = o.bool("preserve_videos", false)?;
        let preserve_audio = o.bool("preserve_audio", false)?;
        let normalize_text = o.bool("normalize_text", false)?;
        let stopword_removal = o.bool("stopword_removal", false)?;
        let stopword_lang = o.string("stopword_lang")?.unwrap_or_else(|| "english".into());
        let external: Vec<(String, String)> =
            o.take::<indexmap::IndexMap<String, String>>("external_metadata")?.unwrap_or_default().into_iter().collect();
        let header_tags: Vec<String> = headers.iter().map(|(t, _)| t.clone()).collect();
        let allowlist = o.strings("allowlist_tags")?.filter(|a| !a.is_empty()).map(|mut a| {
            a.extend(header_tags.iter().cloned());
            a
        });
        let denylist = o
            .strings("denylist_tags")?
            .filter(|d| !d.is_empty())
            .map(|d| d.into_iter().filter(|t| !header_tags.contains(t)).collect());
        let separators = o.strings("separators")?;
        let keep = o.keep_separator(KeepSeparator::Start)?;
        o.ignore(&["preserve_parent_metadata"]);
        o.finish()?;
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_keep_separator(keep);
        let splitter = match separators {
            Some(s) if !s.is_empty() => RecursiveSplitter::new(base, &s, false)?,
            _ => RecursiveSplitter::new(base, &[] as &[&str], false)?,
        };
        Ok(Self {
            sizing,
            headers,
            preserve,
            preserve_links,
            preserve_images,
            preserve_videos,
            preserve_audio,
            normalize_text,
            stopwords: if stopword_removal { Some(load_stopwords(&stopword_lang)?) } else { None },
            external,
            allowlist,
            denylist,
            splitter,
            non_word: PyRegex::new(r"[^\w\s]")?,
            spaces: PyRegex::new(r"\s+")?,
        })
    }

    fn clean(&self, text: &str) -> Result<String> {
        let mut t = text.to_string();
        if self.normalize_text {
            t = t.to_lowercase();
            t = self.non_word.replace_all(&t, "")?;
            t = strip(&self.spaces.replace_all(&t, " ")?).to_string();
        }
        if let Some(sw) = &self.stopwords {
            t = pystr::split_whitespace(&t).into_iter().filter(|w| !sw.contains(*w)).collect::<Vec<_>>().join(" ");
        }
        Ok(t)
    }

    /// `_process_media` e `_process_links`: troca as tags por texto Markdown.
    fn rewrite(&self, node: &mut HElement) {
        for c in node.children.iter_mut() {
            if let HNode::Element(e) = c {
                self.rewrite(e);
                let media = match e.name.as_str() {
                    "img" if self.preserve_images => Some("image"),
                    "video" if self.preserve_videos => Some("video"),
                    "audio" if self.preserve_audio => Some("audio"),
                    _ => None,
                };
                if let Some(kind) = media {
                    let src = e.attr("src").unwrap_or("").to_string();
                    *c = HNode::Element(HElement {
                        name: "media-wrapper".into(),
                        attrs: Vec::new(),
                        children: vec![HNode::Text(format!("![{kind}:{src}]({src})"))],
                    });
                }
            }
        }
        if self.preserve_links {
            for c in node.children.iter_mut() {
                if let HNode::Element(e) = c
                    && e.name == "a"
                {
                    let link = format!("[{}]({})", e.text_stripped(), e.attr("href").unwrap_or(""));
                    *c = HNode::Text(link);
                }
            }
        }
    }

    /// `_filter_tags`: remove o que não estiver na allowlist / o que estiver na denylist.
    /// `html`, `head` e `body` (estrutura que o parser HTML5 sempre cria) nunca são removidos.
    fn filter(&self, node: &mut HElement) {
        let scaffold = |n: &str| matches!(n, "html" | "head" | "body");
        node.children.retain(|c| match c {
            HNode::Element(e) if !scaffold(&e.name) => {
                self.allowlist.as_ref().is_none_or(|a| a.contains(&e.name))
                    && !self.denylist.as_ref().is_some_and(|d| d.contains(&e.name))
            }
            _ => true,
        });
        for c in node.children.iter_mut() {
            if let HNode::Element(e) = c {
                self.filter(e);
            }
        }
    }

    /// `_get_element_text`: texto recursivo (inclui comentários e script, como no Python).
    fn element_text(&self, node: &HNode) -> Result<String> {
        match node {
            HNode::Element(e) => {
                let mut text = String::new();
                for c in &e.children {
                    let ct = self.element_text(c)?;
                    let ct = strip(&ct);
                    if !text.is_empty() && !ct.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(ct);
                }
                self.clean(&text)
            }
            HNode::Text(t) | HNode::Comment(t) => self.clean(t),
        }
    }

    fn flush(&self, st: &mut State) -> Result<()> {
        if st.content.is_empty() {
            return Ok(());
        }
        let content = st.content.join(" ");
        st.sections.extend(self.create_sections(&st.headers, &content, &st.preserved)?);
        st.content.clear();
        st.preserved.clear();
        Ok(())
    }

    fn process(&self, elements: &[&HElement], st: &mut State) -> Result<()> {
        for elem in elements {
            if let Some((_, name)) = self.headers.iter().find(|(t, _)| *t == elem.name) {
                self.flush(st)?;
                st.headers = vec![(name.clone(), elem.text_stripped())];
            } else if self.preserve.contains(&elem.name) {
                let ph = format!("PRESERVED_{}", st.placeholders);
                st.preserved.push((ph.clone(), self.element_text(&HNode::Element((*elem).clone()))?));
                st.content.push(ph);
                st.placeholders += 1;
            } else {
                let children: Vec<&HElement> = elem.child_elements().collect();
                if !children.is_empty() {
                    self.process(&children, st)?;
                    let direct = elem.direct_strings().join(" ");
                    if !direct.is_empty() {
                        st.content.push(self.clean(&direct)?);
                    }
                } else {
                    let content = self.element_text(&HNode::Element((*elem).clone()))?;
                    if !content.is_empty() {
                        st.content.push(content);
                    }
                }
            }
        }
        Ok(())
    }

    fn reinsert(content: &str, preserved: &[(String, String)]) -> String {
        let mut c = content.to_string();
        for (ph, text) in preserved.iter().rev() {
            c = c.replace(ph.as_str(), strip(text));
        }
        c
    }

    fn create_sections(&self, headers: &[(String, String)], content: &str, preserved: &[(String, String)]) -> Result<Vec<Section>> {
        let content = strip(&self.spaces.replace_all(content, " ")?).to_string();
        let mut meta: Vec<(String, String)> = headers.to_vec();
        for (k, v) in &self.external {
            if let Some(slot) = meta.iter_mut().find(|(mk, _)| mk == k) {
                slot.1 = v.clone();
            } else {
                meta.push((k.clone(), v.clone()));
            }
        }
        if pystr::len(&content) <= self.sizing.chunk_size {
            return Ok(vec![Section { content: Self::reinsert(&content, preserved), headers: meta }]);
        }
        let mut out = Vec::new();
        for split in self.splitter.split_text(&content)? {
            let with = Self::reinsert(&split, preserved);
            if !strip(&with).is_empty() {
                out.push(Section { content: strip(&with).to_string(), headers: meta.clone() });
            }
        }
        Ok(out)
    }

    pub fn split_html(&self, html: &str) -> Result<Vec<Section>> {
        let mut doc = crate::core::html::parse(html);
        self.rewrite(&mut doc);
        if self.allowlist.is_some() || self.denylist.is_some() {
            self.filter(&mut doc);
        }
        let mut st = State { sections: Vec::new(), headers: Vec::new(), content: Vec::new(), preserved: Vec::new(), placeholders: 0 };
        let top: Vec<&HElement> = doc.child_elements().collect();
        self.process(&top, &mut st)?;
        self.flush(&mut st)?;
        Ok(st.sections)
    }
}

impl Chunker for ChunkHtmlSemantic {
    crate::sizing_accessors!("html_semantic");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = self.split_html(text)?.iter().map(with_heading).collect();
        enforce_size(&pystr::normalize_chunks(texts), self.sizing)
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        let mut out = Vec::new();
        for s in self.split_html(text)? {
            for c in enforce_size(&pystr::normalize_chunks([with_heading(&s)]), self.sizing)? {
                out.push(Document::with_metadata(c, s.metadata()));
            }
        }
        Ok(out)
    }
}
