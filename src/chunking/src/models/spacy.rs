//! Port do tokenizador do spaCy (`spacy.blank(lang)`) e do componente `sentencizer`, o que a
//! estratégia `spacy` do chatbot usa (`spacy.blank("pt")` + `add_pipe("sentencizer")`).
//!
//! O algoritmo é o do `Tokenizer.explain()`/`_tokenize_affixes` do spaCy 3.8; as regras
//! (prefixos, sufixos, infixos, URL e casos especiais como "Dr." e "etc.") e os caracteres de
//! fim de frase foram extraídos do spaCy e estão em `data/spacy/<lang>.json` (spaCy é MIT).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::core::pyre::PyRegex;
use crate::core::pystr;
use crate::error::{ChunkError, Result};

#[derive(Deserialize)]
struct LangData {
    prefix: String,
    suffix: String,
    infix: String,
    url_match: Option<String>,
    token_match: Option<String>,
    special_cases: HashMap<String, Vec<String>>,
    punct_chars: Vec<String>,
}

pub struct SpacySentencizer {
    prefix: PyRegex,
    suffix: PyRegex,
    infix: PyRegex,
    url_match: Option<PyRegex>,
    token_match: Option<PyRegex>,
    special: HashMap<String, Vec<String>>,
    punct_chars: HashSet<String>,
}

fn data_for(language: &str) -> Option<&'static str> {
    match language {
        "pt" => Some(include_str!("../../data/spacy/pt.json")),
        "en" => Some(include_str!("../../data/spacy/en.json")),
        _ => None,
    }
}

/// Instância compartilhada por idioma (compilar as regex é caro).
pub fn sentencizer(language: &str) -> Result<&'static SpacySentencizer> {
    static PT: OnceLock<std::result::Result<SpacySentencizer, String>> = OnceLock::new();
    static EN: OnceLock<std::result::Result<SpacySentencizer, String>> = OnceLock::new();
    let cell = match language {
        "pt" => &PT,
        "en" => &EN,
        other => return Err(ChunkError::Model(format!("spaCy: idioma '{other}' sem dados de tokenização (pt, en)"))),
    };
    cell.get_or_init(|| SpacySentencizer::load(language).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| ChunkError::Model(e.clone()))
}

/// Um token: posição em bytes no texto original.
#[derive(Debug, Clone, Copy)]
struct Tok {
    start: usize,
    end: usize,
}

impl SpacySentencizer {
    fn load(language: &str) -> Result<Self> {
        let raw = data_for(language).ok_or_else(|| ChunkError::Model(format!("spaCy: idioma '{language}'")))?;
        let d: LangData = serde_json::from_str(raw).map_err(|e| ChunkError::Json(e.to_string()))?;
        Ok(Self {
            prefix: PyRegex::new(&d.prefix)?,
            suffix: PyRegex::new(&d.suffix)?,
            infix: PyRegex::new(&d.infix)?,
            url_match: d.url_match.as_deref().map(PyRegex::new).transpose()?,
            token_match: d.token_match.as_deref().map(PyRegex::new).transpose()?,
            special: d.special_cases,
            punct_chars: d.punct_chars.into_iter().collect(),
        })
    }

    fn token_match(&self, s: &str) -> bool {
        self.token_match.as_ref().is_some_and(|r| r.match_start(s).ok().flatten().is_some())
    }

    fn url_match(&self, s: &str) -> bool {
        self.url_match.as_ref().is_some_and(|r| r.match_start(s).ok().flatten().is_some())
    }

    fn special(&self, s: &str) -> Option<&Vec<String>> {
        self.special.get(s)
    }

    /// Tokeniza um trecho sem espaços (o laço de `Tokenizer.explain()`), devolvendo os
    /// comprimentos (em bytes) dos tokens, em ordem.
    fn split_affixes(&self, span: &str) -> Result<Vec<usize>> {
        let mut tokens: Vec<usize> = Vec::new();
        let mut suffixes: Vec<usize> = Vec::new();
        let mut sub = span;
        let push_special = |tokens: &mut Vec<usize>, parts: &Vec<String>| tokens.extend(parts.iter().map(String::len));
        while !sub.is_empty() {
            if let Some(parts) = self.special(sub) {
                push_special(&mut tokens, parts);
                sub = "";
                continue;
            }
            loop {
                let pre = self.prefix.search(sub)?;
                let suf = self.suffix.search(sub)?;
                if pre.is_none() && suf.is_none() {
                    break;
                }
                if self.token_match(sub) {
                    tokens.push(sub.len());
                    sub = "";
                    break;
                }
                if let Some(parts) = self.special(sub) {
                    push_special(&mut tokens, parts);
                    sub = "";
                    break;
                }
                if let Some(m) = pre {
                    let split = m.end;
                    if split == 0 {
                        break;
                    }
                    tokens.push(split);
                    sub = &sub[split..];
                    if self.special(sub).is_some() {
                        continue;
                    }
                }
                if let Some(m) = self.suffix.search(sub)? {
                    let split = m.start;
                    if split == sub.len() {
                        break;
                    }
                    suffixes.push(sub.len() - split);
                    sub = &sub[..split];
                }
            }
            if sub.is_empty() {
                continue;
            }
            if self.token_match(sub) || self.url_match(sub) {
                tokens.push(sub.len());
            } else if let Some(parts) = self.special(sub) {
                push_special(&mut tokens, parts);
            } else {
                let infixes = self.infix.find_iter(sub)?;
                if infixes.is_empty() {
                    tokens.push(sub.len());
                } else {
                    let mut offset = 0;
                    for m in infixes {
                        if offset == 0 && m.start == 0 {
                            continue;
                        }
                        if m.start > offset {
                            tokens.push(m.start - offset);
                        }
                        if m.end > m.start {
                            tokens.push(m.end - m.start);
                        }
                        offset = m.end;
                    }
                    if offset < sub.len() {
                        tokens.push(sub.len() - offset);
                    }
                }
            }
            sub = "";
        }
        tokens.extend(suffixes.into_iter().rev());
        Ok(tokens)
    }

    /// Tokens do texto, incluindo os de espaço (um `' '` simples vira espaço "colado" no token
    /// anterior, como no spaCy).
    fn tokenize(&self, text: &str) -> Result<Vec<Tok>> {
        let mut toks = Vec::new();
        let mut emit = |span_start: usize, span_end: usize, this: &Self| -> Result<()> {
            let span = &text[span_start..span_end];
            if pystr::is_all_space(span) {
                toks.push(Tok { start: span_start, end: span_end });
                return Ok(());
            }
            let mut pos = span_start;
            for len in this.split_affixes(span)? {
                toks.push(Tok { start: pos, end: pos + len });
                pos += len;
            }
            Ok(())
        };
        let mut chars = text.char_indices().peekable();
        let Some(&(_, first)) = chars.peek() else { return Ok(toks) };
        let mut in_ws = pystr::is_space(first);
        let mut start = 0;
        for (i, c) in text.char_indices() {
            if pystr::is_space(c) != in_ws {
                if start < i {
                    emit(start, i, self)?;
                }
                start = if c == ' ' { i + 1 } else { i };
                in_ws = !in_ws;
            }
        }
        if start < text.len() {
            emit(start, text.len(), self)?;
        }
        Ok(toks)
    }

    /// Frases (posições em bytes) — o `Sentencizer.predict` do spaCy.
    pub fn sentence_spans(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        let toks = self.tokenize(text)?;
        if toks.is_empty() {
            return Ok(Vec::new());
        }
        let mut starts = vec![false; toks.len()];
        starts[0] = true;
        let mut start = 0;
        let mut seen_period = false;
        for (i, t) in toks.iter().enumerate() {
            let s = &text[t.start..t.end];
            let in_punct = self.punct_chars.contains(s);
            let is_punct = !s.is_empty() && s.chars().all(pystr::is_punctuation);
            if seen_period && !is_punct && !in_punct {
                starts[start] = true;
                start = i;
                seen_period = false;
            } else if in_punct {
                seen_period = true;
            }
        }
        if start < toks.len() {
            starts[start] = true;
        }
        let mut spans = Vec::new();
        let mut cur = 0;
        for i in 1..=toks.len() {
            if i == toks.len() || starts[i] {
                spans.push((toks[cur].start, toks[i - 1].end));
                cur = i;
            }
        }
        Ok(spans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portuguese_abbreviations_do_not_split() {
        let s = sentencizer("pt").unwrap();
        let text = "O Dr. Silva chegou às 10h. Ele saiu cedo! Voltou? Sim.";
        let sents: Vec<&str> = s.sentence_spans(text).unwrap().iter().map(|&(a, b)| &text[a..b]).collect();
        assert_eq!(sents, ["O Dr. Silva chegou às 10h.", "Ele saiu cedo!", "Voltou?", "Sim."]);
    }
}
