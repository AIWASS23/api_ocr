//! `javascript` — JavaScript/TypeScript/JSX/TSX/Vue/Svelte (`JSFrameworkTextSplitter`): além
//! dos separadores de JS (`export`, `function`, `const`, `class`, `if`...), corta antes de cada
//! componente encontrado no texto (`<Header`, `<Card`...), na ordem em que aparecem.
//!
//! Opções: `separators` (vêm antes dos de JS), `keep_separator` (`true`), `strip_whitespace`.

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pyre::PyRegex;
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, Separator, TextSplitter};
use crate::error::Result;

const JS_SEPARATORS: [&str; 22] = [
    "\nexport ", " export ", "\nfunction ", "\nasync function ", " async function ", "\nconst ", "\nlet ", "\nvar ",
    "\nclass ", " class ", "\nif ", " if ", "\nfor ", " for ", "\nwhile ", " while ", "\nswitch ", " switch ", "\ncase ",
    " case ", "\ndefault ", " default ",
];

pub struct ChunkJavaScript {
    sizing: Sizing,
    splitter: RecursiveSplitter,
    extra: Vec<String>,
    opening_tags: PyRegex,
}

impl ChunkJavaScript {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let extra = o.strings("separators")?.unwrap_or_default();
        let keep = o.keep_separator(KeepSeparator::Start)?;
        let strip_ws = o.bool("strip_whitespace", true)?;
        o.ignore(&["add_start_index"]);
        o.finish()?;
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_keep_separator(keep).with_strip_whitespace(strip_ws);
        Ok(Self {
            sizing,
            splitter: RecursiveSplitter::new(base, &["\n\n"], false)?,
            extra,
            opening_tags: PyRegex::new(r"<\s*([a-zA-Z0-9]+)[^>]*>")?,
        })
    }

    fn separators_for(&self, text: &str) -> Result<Vec<Separator>> {
        let mut tags: Vec<String> = Vec::new();
        for m in self.opening_tags.find_iter(text)? {
            if let Some(Some((s, e))) = m.groups.first() {
                let t = text[*s..*e].to_string();
                if !tags.contains(&t) {
                    tags.push(t);
                }
            }
        }
        let mut raw: Vec<String> = self.extra.clone();
        raw.extend(JS_SEPARATORS.iter().map(|s| s.to_string()));
        raw.extend(tags.iter().map(|t| format!("<{t}")));
        raw.extend(["<>", "\n\n", "&&\n", "||\n"].map(String::from));
        raw.iter().map(|s| Separator::new(s, false)).collect()
    }
}

impl Chunker for ChunkJavaScript {
    crate::sizing_accessors!("javascript");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let seps = self.separators_for(text)?;
        Ok(normalize_chunks(self.splitter.split_text_with_separators(text, &seps)?))
    }
}
