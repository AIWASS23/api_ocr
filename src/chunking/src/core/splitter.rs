//! Motor de corte do LangChain (`TextSplitter`, `CharacterTextSplitter`,
//! `RecursiveCharacterTextSplitter`), portado linha a linha de `langchain_text_splitters`
//! (`base.py`/`character.py`). Quase todas as estratégias usam um destes dois por baixo.

use std::sync::Arc;

use serde_json::Value;

use crate::core::pyre::{PyRegex, escape};
use crate::core::pystr;
use crate::error::{ChunkError, Result};

/// Função que mede um pedaço de texto (caracteres, tokens...).
pub type LengthFn = Arc<dyn Fn(&str) -> usize + Send + Sync>;

/// `len` do Python (code points).
pub fn char_length() -> LengthFn {
    Arc::new(pystr::len)
}

/// `keep_separator` do LangChain: `False`, `True`/`"start"` ou `"end"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepSeparator {
    #[default]
    No,
    Start,
    End,
}

impl KeepSeparator {
    pub fn is_kept(self) -> bool {
        self != KeepSeparator::No
    }

    /// Aceita `false`, `true`, `"start"` e `"end"` (como no Python).
    pub fn from_value(v: &Value) -> Result<Self> {
        match v {
            Value::Bool(false) => Ok(KeepSeparator::No),
            Value::Bool(true) => Ok(KeepSeparator::Start),
            Value::String(s) if s == "start" => Ok(KeepSeparator::Start),
            Value::String(s) if s == "end" => Ok(KeepSeparator::End),
            other => Err(ChunkError::config(format!("keep_separator must be a bool, 'start' or 'end', got {other}"))),
        }
    }
}

/// Um separador já compilado (literal escapado ou regex no dialeto do Python).
#[derive(Debug, Clone)]
pub struct Separator {
    raw: String,
    regex: Option<PyRegex>,
}

impl Separator {
    pub fn new(raw: &str, is_regex: bool) -> Result<Self> {
        let regex = if raw.is_empty() {
            None
        } else if is_regex {
            Some(PyRegex::new(raw)?)
        } else {
            Some(PyRegex::new(&escape(raw))?)
        };
        Ok(Self { raw: raw.to_string(), regex })
    }

    pub fn raw(&self) -> &str {
        &self.raw
    }

    fn is_found_in(&self, text: &str) -> Result<bool> {
        match &self.regex {
            Some(r) => r.is_match(text),
            None => Ok(true),
        }
    }
}

/// `_split_text_with_regex`: corta e, se pedido, gruda o separador no início ou no fim.
pub fn split_with_separator(text: &str, sep: &Separator, keep: KeepSeparator) -> Result<Vec<String>> {
    let splits: Vec<String> = match &sep.regex {
        None => text.chars().map(String::from).collect(),
        Some(re) if keep.is_kept() => {
            let parts = re.split(text, true)?;
            let n = parts.len();
            let mut splits: Vec<String> = if keep == KeepSeparator::End {
                (0..n.saturating_sub(1)).step_by(2).map(|i| format!("{}{}", parts[i], parts[i + 1])).collect()
            } else {
                (1..n).step_by(2).filter(|&i| i + 1 < n).map(|i| format!("{}{}", parts[i], parts[i + 1])).collect()
            };
            if n % 2 == 0 {
                splits.push(parts[n - 1].to_string());
            }
            if keep == KeepSeparator::End {
                splits.push(parts[n - 1].to_string());
            } else {
                splits.insert(0, parts[0].to_string());
            }
            splits
        }
        Some(re) => re.split(text, false)?.into_iter().map(str::to_string).collect(),
    };
    Ok(splits.into_iter().filter(|s| !s.is_empty()).collect())
}

/// Núcleo do `TextSplitter`: tamanho, sobreposição, medida, separador e `strip_whitespace`.
#[derive(Clone)]
pub struct TextSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub length: LengthFn,
    pub keep_separator: KeepSeparator,
    pub strip_whitespace: bool,
}

impl std::fmt::Debug for TextSplitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextSplitter")
            .field("chunk_size", &self.chunk_size)
            .field("chunk_overlap", &self.chunk_overlap)
            .field("keep_separator", &self.keep_separator)
            .finish_non_exhaustive()
    }
}

impl TextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self> {
        if chunk_size == 0 {
            return Err(ChunkError::config(format!("chunk_size must be > 0, got {chunk_size}")));
        }
        if chunk_overlap > chunk_size {
            return Err(ChunkError::config(format!(
                "Got a larger chunk overlap ({chunk_overlap}) than chunk size ({chunk_size}), should be smaller."
            )));
        }
        Ok(Self { chunk_size, chunk_overlap, length: char_length(), keep_separator: KeepSeparator::No, strip_whitespace: true })
    }

    pub fn with_length(mut self, length: LengthFn) -> Self {
        self.length = length;
        self
    }

    pub fn with_keep_separator(mut self, keep: KeepSeparator) -> Self {
        self.keep_separator = keep;
        self
    }

    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    fn join_docs(&self, docs: &[&str], separator: &str) -> Option<String> {
        let text = docs.join(separator);
        let text = if self.strip_whitespace { pystr::strip(&text).to_string() } else { text };
        (!text.is_empty()).then_some(text)
    }

    /// `_merge_splits`: junta pedaços até `chunk_size`, recuando para manter `chunk_overlap`.
    pub fn merge_splits(&self, splits: &[impl AsRef<str>], separator: &str) -> Vec<String> {
        let len = &self.length;
        let separator_len = len(separator);
        let mut docs = Vec::new();
        let mut current: std::collections::VecDeque<&str> = std::collections::VecDeque::new();
        let mut total: usize = 0;
        for d in splits {
            let d = d.as_ref();
            let len_ = len(d);
            let sep_if = |cur: &std::collections::VecDeque<&str>| if cur.is_empty() { 0 } else { separator_len };
            if total + len_ + sep_if(&current) > self.chunk_size
                && !current.is_empty() {
                    let parts: Vec<&str> = current.iter().copied().collect();
                    if let Some(doc) = self.join_docs(&parts, separator) {
                        docs.push(doc);
                    }
                    while total > self.chunk_overlap || (total + len_ + sep_if(&current) > self.chunk_size && total > 0) {
                        let Some(first) = current.front().copied() else { break };
                        let sep = if current.len() > 1 { separator_len } else { 0 };
                        total = total.saturating_sub(len(first) + sep);
                        current.pop_front();
                    }
                }
            current.push_back(d);
            total += len_ + if current.len() > 1 { separator_len } else { 0 };
        }
        let parts: Vec<&str> = current.iter().copied().collect();
        if let Some(doc) = self.join_docs(&parts, separator) {
            docs.push(doc);
        }
        docs
    }
}

/// `CharacterTextSplitter`.
#[derive(Debug, Clone)]
pub struct CharacterSplitter {
    pub base: TextSplitter,
    separator: Separator,
    is_regex: bool,
}

impl CharacterSplitter {
    pub fn new(base: TextSplitter, separator: &str, is_separator_regex: bool) -> Result<Self> {
        Ok(Self { base, separator: Separator::new(separator, is_separator_regex)?, is_regex: is_separator_regex })
    }

    pub fn split_text(&self, text: &str) -> Result<Vec<String>> {
        let splits = split_with_separator(text, &self.separator, self.base.keep_separator)?;
        let raw = self.separator.raw();
        let is_lookaround = self.is_regex && ["(?=", "(?<!", "(?<=", "(?!"].iter().any(|p| raw.starts_with(p));
        let merge_sep = if self.base.keep_separator.is_kept() || is_lookaround { "" } else { raw };
        Ok(self.base.merge_splits(&splits, merge_sep))
    }
}

/// `RecursiveCharacterTextSplitter`.
#[derive(Debug, Clone)]
pub struct RecursiveSplitter {
    pub base: TextSplitter,
    separators: Vec<Separator>,
}

pub const DEFAULT_SEPARATORS: [&str; 4] = ["\n\n", "\n", " ", ""];

impl RecursiveSplitter {
    /// `separators` vazio usa o padrão `["\n\n", "\n", " ", ""]`.
    pub fn new(base: TextSplitter, separators: &[impl AsRef<str>], is_separator_regex: bool) -> Result<Self> {
        let raw: Vec<&str> = if separators.is_empty() {
            DEFAULT_SEPARATORS.to_vec()
        } else {
            separators.iter().map(AsRef::as_ref).collect()
        };
        let separators = raw.iter().map(|s| Separator::new(s, is_separator_regex)).collect::<Result<_>>()?;
        Ok(Self { base, separators })
    }

    /// `from_language`: separadores da linguagem, tratados como regex.
    pub fn from_language(base: TextSplitter, language: &str) -> Result<Self> {
        let seps = crate::core::languages::separators_for(language)?;
        Self::new(base, &seps, true)
    }

    pub fn split_text(&self, text: &str) -> Result<Vec<String>> {
        self.split_with(text, &self.separators)
    }

    /// Variante com separadores calculados na hora (ex.: componentes JSX encontrados no texto).
    pub fn split_text_with_separators(&self, text: &str, separators: &[Separator]) -> Result<Vec<String>> {
        self.split_with(text, separators)
    }

    fn split_with(&self, text: &str, separators: &[Separator]) -> Result<Vec<String>> {
        let mut final_chunks = Vec::new();
        let Some(mut separator) = separators.last() else {
            return Ok(final_chunks);
        };
        let mut new_separators: &[Separator] = &[];
        for (i, s) in separators.iter().enumerate() {
            if s.raw().is_empty() {
                separator = s;
                break;
            }
            if s.is_found_in(text)? {
                separator = s;
                new_separators = &separators[i + 1..];
                break;
            }
        }
        let splits = split_with_separator(text, separator, self.base.keep_separator)?;
        let merge_sep = if self.base.keep_separator.is_kept() { "" } else { separator.raw() };
        let mut good: Vec<String> = Vec::new();
        for s in splits {
            if (self.base.length)(&s) < self.base.chunk_size {
                good.push(s);
            } else {
                if !good.is_empty() {
                    final_chunks.extend(self.base.merge_splits(&good, merge_sep));
                    good.clear();
                }
                if new_separators.is_empty() {
                    final_chunks.push(s);
                } else {
                    final_chunks.extend(self.split_with(&s, new_separators)?);
                }
            }
        }
        if !good.is_empty() {
            final_chunks.extend(self.base.merge_splits(&good, merge_sep));
        }
        Ok(final_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_separator_end_and_start() {
        let sep = Separator::new(". ", false).unwrap();
        assert_eq!(split_with_separator("a. b. c", &sep, KeepSeparator::End).unwrap(), ["a. ", "b. ", "c"]);
        assert_eq!(split_with_separator("a. b. c", &sep, KeepSeparator::Start).unwrap(), ["a", ". b", ". c"]);
        assert_eq!(split_with_separator("a. b", &sep, KeepSeparator::No).unwrap(), ["a", "b"]);
    }

    #[test]
    fn merge_with_overlap() {
        let s = TextSplitter::new(10, 4).unwrap();
        let parts = ["aaa", "bbb", "ccc", "ddd"];
        assert_eq!(s.merge_splits(&parts, " "), ["aaa bbb", "bbb ccc", "ccc ddd"]);
    }

    #[test]
    fn recursive_default() {
        let r = RecursiveSplitter::new(TextSplitter::new(10, 0).unwrap(), &[] as &[&str], false).unwrap();
        assert_eq!(r.split_text("hello world foo bar").unwrap(), ["hello", "world foo", "bar"]);
    }
}
