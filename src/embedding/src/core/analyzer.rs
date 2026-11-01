//! O analisador de texto dos vetorizadores do scikit-learn (`build_analyzer`): minúsculas,
//! remoção de acentos, tokens pelo `token_pattern`, stop words e n-gramas de palavras ou de
//! caracteres (`analyzer`: `word`, `char`, `char_wb`). Usado por `tfidf`, `hashing`, `bm25` e
//! `lsa`, com as mesmas opções e o mesmo resultado do Python.

use std::collections::HashSet;

use chunking::core::pyre::PyRegex;
use chunking::core::pystr;
use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

use crate::core::options::Options;
use crate::core::stop_words::ENGLISH_STOP_WORDS;
use crate::error::{EmbeddingError, Result};

/// Padrão do scikit-learn: palavras de 2 ou mais caracteres.
pub const DEFAULT_TOKEN_PATTERN: &str = r"(?u)\b\w\w+\b";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Word,
    Char,
    CharWb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StripAccents {
    Ascii,
    Unicode,
}

pub struct Analyzer {
    kind: Kind,
    lowercase: bool,
    strip_accents: Option<StripAccents>,
    /// `None`: o padrão do scikit-learn, avaliado nativamente com o `\w` do Python.
    token_pattern: Option<PyRegex>,
    stop_words: Option<HashSet<String>>,
    ngram_range: (usize, usize),
}

impl Default for Analyzer {
    fn default() -> Self {
        Self { kind: Kind::Word, lowercase: true, strip_accents: None, token_pattern: None, stop_words: None, ngram_range: (1, 1) }
    }
}

/// `\w` do `re` do Python para `str`: `str.isalnum()` ou `_`.
pub fn is_word_char(c: char) -> bool {
    if c == '_' || c.is_ascii_alphanumeric() {
        return true;
    }
    if c.is_ascii() {
        return false;
    }
    use GeneralCategory::*;
    matches!(
        get_general_category(c),
        UppercaseLetter | LowercaseLetter | TitlecaseLetter | ModifierLetter | OtherLetter | DecimalNumber | LetterNumber | OtherNumber
    )
}

/// `strip_accents_ascii`: NFKD e só o que for ASCII.
pub fn strip_accents_ascii(s: &str) -> String {
    s.nfkd().filter(char::is_ascii).collect()
}

/// `strip_accents_unicode`: NFKD sem as marcas combinantes.
pub fn strip_accents_unicode(s: &str) -> String {
    if s.is_ascii() {
        return s.to_string();
    }
    s.nfkd().filter(|&c| canonical_combining_class(c) == 0).collect()
}

/// `re.sub(r"\s\s+", " ", text)`.
fn collapse_whitespace(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if pystr::is_space(chars[i]) && i + 1 < chars.len() && pystr::is_space(chars[i + 1]) {
            while i < chars.len() && pystr::is_space(chars[i]) {
                i += 1;
            }
            out.push(' ');
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

impl Analyzer {
    /// Retira de `options` as opções do analisador (as mesmas do `CountVectorizer`).
    pub fn from_options(o: &mut Options) -> Result<Self> {
        let kind = match o.string("analyzer")?.as_deref() {
            None | Some("word") => Kind::Word,
            Some("char") => Kind::Char,
            Some("char_wb") => Kind::CharWb,
            Some(other) => return Err(EmbeddingError::config(format!("analyzer must be 'word', 'char' or 'char_wb', got '{other}'"))),
        };
        if o.has("preprocessor") || o.has("tokenizer") {
            return Err(EmbeddingError::config("preprocessor/tokenizer callables are not supported; use token_pattern"));
        }
        let lowercase = o.bool("lowercase", true)?;
        let strip_accents = match o.string("strip_accents")?.as_deref() {
            None => None,
            Some("ascii") => Some(StripAccents::Ascii),
            Some("unicode") => Some(StripAccents::Unicode),
            Some(other) => return Err(EmbeddingError::config(format!("Invalid value for 'strip_accents': {other}"))),
        };
        let token_pattern = match o.string("token_pattern")? {
            None => None,
            Some(p) if p == DEFAULT_TOKEN_PATTERN => None,
            Some(p) => {
                let re = PyRegex::new(&p).map_err(|e| EmbeddingError::config(format!("token_pattern: {e}")))?;
                if group_count(&p) > 1 {
                    return Err(EmbeddingError::config(format!(
                        "More than 1 capturing group in token pattern. Only a single group should be captured. ({p})"
                    )));
                }
                Some(re)
            }
        };
        let stop_words = match o.raw("stop_words") {
            None => None,
            Some(serde_json::Value::String(s)) if s == "english" => Some(ENGLISH_STOP_WORDS.iter().map(|w| w.to_string()).collect()),
            Some(serde_json::Value::String(s)) => return Err(EmbeddingError::config(format!("not a built-in stop list: {s}"))),
            Some(serde_json::Value::Array(a)) => Some(
                a.into_iter()
                    .map(|v| v.as_str().map(str::to_string).ok_or_else(|| EmbeddingError::config("stop_words must be 'english' or a list of strings")))
                    .collect::<Result<HashSet<_>>>()?,
            ),
            Some(_) => return Err(EmbeddingError::config("stop_words must be 'english' or a list of strings")),
        };
        let ngram_range = match o.take::<(usize, usize)>("ngram_range")? {
            None => (1, 1),
            Some((a, b)) if a >= 1 && a <= b => (a, b),
            Some((a, b)) => {
                return Err(EmbeddingError::config(format!(
                    "Invalid value for ngram_range=({a}, {b}) lower boundary larger than the upper boundary."
                )));
            }
        };
        Ok(Self { kind, lowercase, strip_accents, token_pattern, stop_words, ngram_range })
    }

    fn preprocess(&self, doc: &str) -> String {
        let doc = if self.lowercase { doc.to_lowercase() } else { doc.to_string() };
        match self.strip_accents {
            None => doc,
            Some(StripAccents::Ascii) => strip_accents_ascii(&doc),
            Some(StripAccents::Unicode) => strip_accents_unicode(&doc),
        }
    }

    fn tokenize(&self, doc: &str) -> Vec<String> {
        match &self.token_pattern {
            None => {
                // `\b\w\w+\b` = toda sequência máxima de 2+ caracteres de palavra.
                let mut out = Vec::new();
                let mut cur = String::new();
                let mut n = 0;
                for c in doc.chars() {
                    if is_word_char(c) {
                        cur.push(c);
                        n += 1;
                    } else {
                        if n >= 2 {
                            out.push(std::mem::take(&mut cur));
                        }
                        cur.clear();
                        n = 0;
                    }
                }
                if n >= 2 {
                    out.push(cur);
                }
                out
            }
            Some(re) => re
                .find_iter(doc)
                .unwrap_or_default()
                .into_iter()
                .map(|m| match m.groups.first() {
                    // Com um grupo, `findall` devolve o grupo (vazio se não participou).
                    Some(g) => g.map(|(s, e)| doc[s..e].to_string()).unwrap_or_default(),
                    None => doc[m.start..m.end].to_string(),
                })
                .collect(),
        }
    }

    fn word_ngrams(&self, mut tokens: Vec<String>) -> Vec<String> {
        if let Some(stop) = &self.stop_words {
            tokens.retain(|w| !stop.contains(w));
        }
        let (mut min_n, max_n) = self.ngram_range;
        if max_n == 1 {
            return tokens;
        }
        let original = tokens;
        let mut out = if min_n == 1 {
            min_n += 1;
            original.clone()
        } else {
            Vec::new()
        };
        let n_original = original.len();
        for n in min_n..=max_n.min(n_original) {
            for i in 0..=(n_original - n) {
                out.push(original[i..i + n].join(" "));
            }
        }
        out
    }

    fn char_ngrams(&self, text: &str) -> Vec<String> {
        let text: Vec<char> = collapse_whitespace(text).chars().collect();
        let (mut min_n, max_n) = self.ngram_range;
        let mut out: Vec<String> = Vec::new();
        if min_n == 1 {
            out.extend(text.iter().map(|c| c.to_string()));
            min_n += 1;
        }
        for n in min_n..=max_n.min(text.len()) {
            for i in 0..=(text.len() - n) {
                out.push(text[i..i + n].iter().collect());
            }
        }
        out
    }

    fn char_wb_ngrams(&self, text: &str) -> Vec<String> {
        let text = collapse_whitespace(text);
        let (min_n, max_n) = self.ngram_range;
        let mut out = Vec::new();
        for word in pystr::split_whitespace(&text) {
            let w: Vec<char> = std::iter::once(' ').chain(word.chars()).chain(std::iter::once(' ')).collect();
            for n in min_n..=max_n {
                let mut offset = 0;
                out.push(w[offset..(offset + n).min(w.len())].iter().collect());
                while offset + n < w.len() {
                    offset += 1;
                    out.push(w[offset..offset + n].iter().collect());
                }
                if offset == 0 {
                    // Palavra curta (menor que n): conta uma vez só.
                    break;
                }
            }
        }
        out
    }

    /// Os termos (tokens ou n-gramas) de um documento, na ordem do scikit-learn.
    pub fn analyze(&self, doc: &str) -> Vec<String> {
        let doc = self.preprocess(doc);
        match self.kind {
            Kind::Word => self.word_ngrams(self.tokenize(&doc)),
            Kind::Char => self.char_ngrams(&doc),
            Kind::CharWb => self.char_wb_ngrams(&doc),
        }
    }
}

/// Grupos de captura de um padrão (ignora `(?...)` e parênteses escapados/em classes).
fn group_count(pattern: &str) -> usize {
    let b = pattern.as_bytes();
    let (mut i, mut n, mut in_class) = (0, 0, false);
    while i < b.len() {
        match b[i] {
            b'\\' => i += 1,
            b'[' => in_class = true,
            b']' => in_class = false,
            b'(' if !in_class && b.get(i + 1) != Some(&b'?') => n += 1,
            b'(' if !in_class && b.get(i + 1) == Some(&b'?') && b.get(i + 2) == Some(&b'P') && b.get(i + 3) == Some(&b'<') => n += 1,
            _ => {}
        }
        i += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn analyzer(options: serde_json::Value) -> Analyzer {
        let map = options.as_object().cloned().unwrap_or_default();
        let mut o = Options::new("tfidf", &map);
        let a = Analyzer::from_options(&mut o).unwrap();
        o.finish().unwrap();
        a
    }

    #[test]
    fn word_tokens_and_ngrams_like_sklearn() {
        let a = analyzer(json!({}));
        assert_eq!(a.analyze("Ação é a_b 42 x Fortaleza-CE!"), vec!["ação", "a_b", "42", "fortaleza", "ce"]);
        let a = analyzer(json!({"ngram_range": [1, 2], "stop_words": ["de"]}));
        assert_eq!(a.analyze("Busca de documentos rápida"), vec!["busca", "documentos", "rápida", "busca documentos", "documentos rápida"]);
        let a = analyzer(json!({"strip_accents": "unicode"}));
        assert_eq!(a.analyze("Ação rápida"), vec!["acao", "rapida"]);
        let a = analyzer(json!({"strip_accents": "ascii", "lowercase": false}));
        assert_eq!(a.analyze("Ação ½"), vec!["Acao", "12"]);
    }

    #[test]
    fn char_analyzers() {
        let a = analyzer(json!({"analyzer": "char", "ngram_range": [2, 3]}));
        assert_eq!(a.analyze("ab  c"), vec!["ab", "b ", " c", "ab ", "b c"]);
        let a = analyzer(json!({"analyzer": "char_wb", "ngram_range": [2, 3]}));
        assert_eq!(a.analyze("ab c"), vec![" a", "ab", "b ", " ab", "ab ", " c", "c ", " c "]);
    }

    #[test]
    fn custom_pattern_with_one_group_and_invalid_options() {
        let a = analyzer(json!({"token_pattern": r"#(\w+)"}));
        assert_eq!(a.analyze("#rust e #rag"), vec!["rust", "rag"]);
        let map = json!({"token_pattern": r"(a)(b)"}).as_object().cloned().unwrap();
        assert!(Analyzer::from_options(&mut Options::new("tfidf", &map)).is_err());
        let map = json!({"ngram_range": [3, 1]}).as_object().cloned().unwrap();
        assert!(Analyzer::from_options(&mut Options::new("tfidf", &map)).is_err());
    }
}
