//! Port do `PunktSentenceTokenizer` do NLTK (só a tokenização; o treino não é necessário).
//!
//! Os parâmetros pré-treinados vêm do pacote `punkt_tab` do NLTK (`abbrev_types.txt`,
//! `collocations.tab`, `sent_starters.txt`, `ortho_context.tab`), procurado nos mesmos lugares
//! que o NLTK procura (`NLTK_DATA`, `~/nltk_data`, `/usr/share/nltk_data`...). É o que o
//! `nltk.sent_tokenize(text, language)` do chatbot usa.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use fancy_regex::Regex;

use crate::core::pystr;
use crate::error::{ChunkError, Result};

const ORTHO_BEG_UC: u32 = 1 << 1;
const ORTHO_MID_UC: u32 = 1 << 2;
const ORTHO_UNK_UC: u32 = 1 << 3;
const ORTHO_BEG_LC: u32 = 1 << 4;
const ORTHO_MID_LC: u32 = 1 << 5;
const ORTHO_UNK_LC: u32 = 1 << 6;
const ORTHO_UC: u32 = ORTHO_BEG_UC + ORTHO_MID_UC + ORTHO_UNK_UC;
const ORTHO_LC: u32 = ORTHO_BEG_LC + ORTHO_MID_LC + ORTHO_UNK_LC;

#[derive(Debug, Default, Clone)]
pub struct PunktParameters {
    pub abbrev_types: HashSet<String>,
    pub collocations: HashSet<(String, String)>,
    pub sent_starters: HashSet<String>,
    pub ortho_context: HashMap<String, u32>,
}

/// Diretórios onde o NLTK procura dados (`nltk.data.path`).
pub fn nltk_data_paths(extra: Option<&Path>) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = extra.map(Path::to_path_buf).into_iter().collect();
    if let Ok(env) = std::env::var("NLTK_DATA") {
        paths.extend(std::env::split_paths(&env));
    }
    if let Some(home) = std::env::var_os("HOME") {
        paths.push(PathBuf::from(home).join("nltk_data"));
    }
    for p in ["/usr/share/nltk_data", "/usr/local/share/nltk_data", "/usr/lib/nltk_data", "/usr/local/lib/nltk_data"] {
        paths.push(PathBuf::from(p));
    }
    paths
}

impl PunktParameters {
    /// Carrega `tokenizers/punkt_tab/<language>/` do primeiro diretório que tiver os arquivos.
    pub fn load(language: &str, extra_path: Option<&Path>) -> Result<Self> {
        let dir = nltk_data_paths(extra_path)
            .into_iter()
            .map(|p| p.join("tokenizers/punkt_tab").join(language))
            .find(|d| d.join("abbrev_types.txt").is_file())
            .ok_or_else(|| ChunkError::Model(format!("Resource punkt_tab not found (tokenizers/punkt_tab/{language}/)")))?;
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).map_err(|e| ChunkError::Model(format!("{f}: {e}")));
        let lines = |s: String| s.split('\n').filter(|l| !l.is_empty()).map(str::to_string).collect::<Vec<_>>();
        let mut p = PunktParameters {
            abbrev_types: lines(read("abbrev_types.txt")?).into_iter().collect(),
            sent_starters: lines(read("sent_starters.txt")?).into_iter().collect(),
            ..PunktParameters::default()
        };
        for l in lines(read("collocations.tab")?) {
            if let Some((a, b)) = l.split_once('\t') {
                p.collocations.insert((a.to_string(), b.to_string()));
            }
        }
        for l in lines(read("ortho_context.tab")?) {
            if let Some((a, b)) = l.split_once('\t') {
                p.ortho_context.insert(a.to_string(), b.trim().parse().unwrap_or(0));
            }
        }
        Ok(p)
    }

    fn ortho(&self, typ: &str) -> u32 {
        self.ortho_context.get(typ).copied().unwrap_or(0)
    }
}

/// `[^\W\d]`: caractere de palavra que não é dígito decimal.
fn is_letterish(c: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};
    (c.is_alphanumeric() || c == '_') && get_general_category(c) != GeneralCategory::DecimalNumber
}

#[derive(Debug, Clone)]
struct Token {
    tok: String,
    typ: String,
    period_final: bool,
    sentbreak: bool,
    abbr: bool,
    ellipsis: bool,
}

impl Token {
    fn new(tok: &str) -> Self {
        let lower = tok.to_lowercase();
        let typ = if is_numeric_token(&lower) { "##number##".to_string() } else { lower };
        Self { tok: tok.to_string(), typ, period_final: tok.ends_with('.'), sentbreak: false, abbr: false, ellipsis: false }
    }

    fn type_no_period(&self) -> &str {
        if self.typ.chars().count() > 1 && self.typ.ends_with('.') { &self.typ[..self.typ.len() - 1] } else { &self.typ }
    }

    fn type_no_sentperiod(&self) -> &str {
        if self.sentbreak { self.type_no_period() } else { &self.typ }
    }

    fn first_upper(&self) -> bool {
        pystr::first_is_upper(&self.tok)
    }

    fn first_lower(&self) -> bool {
        pystr::first_is_lower(&self.tok)
    }

    /// `\.\.+$` com `match`.
    fn is_ellipsis(&self) -> bool {
        self.tok.len() >= 2 && self.tok.chars().all(|c| c == '.')
    }

    /// `[^\W\d]\.$` com `match`.
    fn is_initial(&self) -> bool {
        let mut it = self.tok.chars();
        matches!((it.next(), it.next(), it.next()), (Some(c), Some('.'), None) if is_letterish(c))
    }
}

/// `\d` do Python: dígito decimal Unicode (categoria Nd).
fn is_decimal(c: char) -> bool {
    use unicode_general_category::{GeneralCategory, get_general_category};
    get_general_category(c) == GeneralCategory::DecimalNumber
}

/// `^-?[\.,]?\d[\d,\.-]*\.?$` sobre o token em minúsculas.
fn is_numeric_token(s: &str) -> bool {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    if c.get(i) == Some(&'-') {
        i += 1;
    }
    if matches!(c.get(i), Some('.' | ',')) {
        i += 1;
    }
    if !c.get(i).is_some_and(|ch| ch.is_ascii_digit() || ch.is_numeric() && ch.is_ascii_digit()) {
        return false;
    }
    i += 1;
    while i < c.len() && (is_decimal(c[i]) || matches!(c[i], ',' | '.' | '-')) {
        i += 1;
    }
    if i < c.len() && c[i] == '.' {
        i += 1;
    }
    i == c.len()
}

pub struct PunktTokenizer {
    params: PunktParameters,
    word_re: Regex,
    period_context_re: Regex,
    realign_re: Regex,
}

const NON_WORD: &str = r#"(?:[)";}\]*:@'({\[‘’“”«»?!])"#;
const MULTI_CHAR: &str = r"(?:-{2,}|\.{2,}|(?:\.\s){2,}\.)";
const WORD_START: &str = r#"[^("`{\[:;&#*@)}\]\-,]"#;

impl PunktTokenizer {
    pub fn new(params: PunktParameters) -> Result<Self> {
        let rx = |p: String| Regex::new(&p).map_err(|e| ChunkError::Regex(e.to_string()));
        let word = format!(
            r"({mc}|(?={ws})\S+?(?=\s|\z|{nw}|{mc}|,(?=\z|\s|{nw}|{mc}))|\S)",
            mc = MULTI_CHAR,
            ws = WORD_START,
            nw = NON_WORD
        );
        let period = format!(r"[\.\?!](?=(?P<after_tok>{nw}|\s+(?P<next_tok>\S+)))", nw = NON_WORD);
        let realign = r#"(?m)["')\]}‘’“”«»]+?(?:\s+|(?=--)|$)"#.to_string();
        Ok(Self { params, word_re: rx(word)?, period_context_re: rx(period)?, realign_re: rx(realign)? })
    }

    pub fn load(language: &str, extra_path: Option<&Path>) -> Result<Self> {
        Self::new(PunktParameters::load(language, extra_path)?)
    }

    fn word_tokenize<'a>(&self, line: &'a str) -> Vec<&'a str> {
        let mut out = Vec::new();
        let mut pos = 0;
        while pos < line.len() {
            match self.word_re.find_from_pos(line, pos) {
                Ok(Some(m)) if m.end() > m.start() => {
                    out.push(m.as_str());
                    pos = m.end();
                }
                _ => break,
            }
        }
        out
    }

    fn tokenize_words(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        for line in text.split('\n') {
            if !pystr::strip(line).is_empty() {
                out.extend(self.word_tokenize(line).into_iter().map(Token::new));
            }
        }
        out
    }

    fn first_pass(&self, t: &mut Token) {
        let tok = t.tok.as_str();
        if matches!(tok, "." | "?" | "!") {
            t.sentbreak = true;
        } else if t.is_ellipsis() {
            t.ellipsis = true;
        } else if t.period_final && !tok.ends_with("..") {
            let stem = tok[..tok.len() - 1].to_lowercase();
            let last = stem.rsplit('-').next().unwrap_or("");
            if self.params.abbrev_types.contains(&stem) || self.params.abbrev_types.contains(last) {
                t.abbr = true;
            } else {
                t.sentbreak = true;
            }
        }
    }

    /// `True` / `False` / `"unknown"` → `Some(true)` / `Some(false)` / `None`.
    fn ortho_heuristic(&self, t: &Token) -> Option<bool> {
        if matches!(t.tok.as_str(), ";" | ":" | "," | "." | "!" | "?") {
            return Some(false);
        }
        let ctx = self.params.ortho(t.type_no_sentperiod());
        if t.first_upper() && (ctx & ORTHO_LC) != 0 && (ctx & ORTHO_MID_UC) == 0 {
            return Some(true);
        }
        if t.first_lower() && ((ctx & ORTHO_UC) != 0 || (ctx & ORTHO_BEG_LC) == 0) {
            return Some(false);
        }
        None
    }

    fn second_pass(&self, t1: &mut Token, t2: &Token) {
        if !t1.period_final {
            return;
        }
        let typ = t1.type_no_period().to_string();
        let next_typ = t2.type_no_sentperiod().to_string();
        let initial = t1.is_initial();
        if self.params.collocations.contains(&(typ.clone(), next_typ.clone())) {
            t1.sentbreak = false;
            t1.abbr = true;
            return;
        }
        if (t1.abbr || t1.ellipsis) && !initial {
            if self.ortho_heuristic(t2) == Some(true) {
                t1.sentbreak = true;
                return;
            }
            if t2.first_upper() && self.params.sent_starters.contains(&next_typ) {
                t1.sentbreak = true;
                return;
            }
        }
        if initial || typ == "##number##" {
            let starter = self.ortho_heuristic(t2);
            if starter == Some(false) {
                t1.sentbreak = false;
                t1.abbr = true;
                return;
            }
            if starter.is_none() && initial && t2.first_upper() && (self.params.ortho(&next_typ) & ORTHO_LC) == 0 {
                t1.sentbreak = false;
                t1.abbr = true;
            }
        }
    }

    fn annotate(&self, mut tokens: Vec<Token>) -> Vec<Token> {
        for t in tokens.iter_mut() {
            self.first_pass(t);
        }
        for i in 0..tokens.len() {
            if i + 1 < tokens.len() {
                let (a, b) = tokens.split_at_mut(i + 1);
                self.second_pass(&mut a[i], &b[0]);
            }
        }
        tokens
    }

    fn text_contains_sentbreak(&self, text: &str) -> bool {
        let tokens = self.annotate(self.tokenize_words(text));
        let mut found = false;
        for t in tokens {
            if found {
                return true;
            }
            if t.sentbreak {
                found = true;
            }
        }
        false
    }

    /// Índice do último caractere de `string.whitespace` (`' \t\n\r\x0b\x0c'`), ou 0.
    fn last_whitespace_index(text: &str) -> usize {
        text.char_indices()
            .rev()
            .find(|(_, c)| matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{0b}' | '\u{0c}'))
            .map_or(0, |(i, _)| i)
    }

    fn slices(&self, text: &str) -> Vec<(usize, usize)> {
        // (match_start, match_end, next_tok_start, context)
        let mut contexts: Vec<(usize, usize, Option<usize>, String)> = Vec::new();
        let mut prev_slice = (0usize, 0usize);
        let mut prev_match: Option<(usize, usize, Option<usize>, usize)> = None; // start, end, next_tok, after_end
        let mut pos = 0;
        let mut matches = Vec::new();
        while pos <= text.len() {
            let Ok(Some(c)) = self.period_context_re.captures_from_pos(text, pos) else { break };
            let m = c.get(0).expect("grupo 0");
            let after_end = c.name("after_tok").map_or(m.end(), |a| a.end());
            let next_tok = c.name("next_tok").map(|n| n.start());
            matches.push((m.start(), m.end(), next_tok, after_end));
            pos = if m.end() > m.start() { m.end() } else { m.end() + 1 };
        }
        let ctx = |slice: (usize, usize), m: &(usize, usize, Option<usize>, usize)| {
            format!("{}{}", &text[slice.0..slice.1], &text[m.0..m.3])
        };
        for m in &matches {
            let before = &text[prev_slice.1..m.0];
            let mut idx = Self::last_whitespace_index(before);
            idx = if idx != 0 { idx + prev_slice.1 + 1 } else { prev_slice.0 };
            let word_slice = (idx, m.0);
            if let Some(pm) = prev_match
                && prev_slice.1 <= word_slice.0
            {
                contexts.push((pm.0, pm.1, pm.2, ctx(prev_slice, &pm)));
            }
            prev_match = Some(*m);
            prev_slice = word_slice;
        }
        if let Some(pm) = prev_match {
            contexts.push((pm.0, pm.1, pm.2, ctx(prev_slice, &pm)));
        }

        let mut out = Vec::new();
        let mut last_break = 0;
        for (_start, end, next_tok, context) in contexts {
            if self.text_contains_sentbreak(&context) {
                out.push((last_break, end));
                last_break = next_tok.unwrap_or(end);
            }
        }
        out.push((last_break, pystr::rstrip(text).len()));
        out
    }

    fn realign(&self, text: &str, slices: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut realign = 0;
        for i in 0..slices.len() {
            let s1 = (slices[i].0 + realign, slices[i].1);
            let Some(&s2) = slices.get(i + 1) else {
                if s1.0 < s1.1 {
                    out.push(s1);
                }
                continue;
            };
            let sub = &text[s2.0..s2.1.max(s2.0)];
            match self.realign_re.find(sub) {
                Ok(Some(m)) if m.start() == 0 => {
                    out.push((s1.0, s2.0 + pystr::rstrip(m.as_str()).len()));
                    realign = m.end();
                }
                _ => {
                    realign = 0;
                    if s1.0 < s1.1 {
                        out.push(s1);
                    }
                }
            }
        }
        out
    }

    /// `span_tokenize` (com realinhamento), em bytes.
    pub fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let slices = self.slices(text);
        self.realign(text, slices)
    }

    /// `sent_tokenize`.
    pub fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        self.span_tokenize(text).into_iter().map(|(s, e)| &text[s..e.max(s)]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_types() {
        assert!(is_numeric_token("1.234,56"));
        assert!(is_numeric_token("-3."));
        assert!(!is_numeric_token("a1"));
    }

    #[test]
    fn splits_with_abbreviations() {
        let mut p = PunktParameters::default();
        p.abbrev_types.insert("sr".into());
        let t = PunktTokenizer::new(p).unwrap();
        let s = t.tokenize("O Sr. Silva chegou. Ele saiu cedo! Voltou?");
        assert_eq!(s, ["O Sr. Silva chegou.", "Ele saiu cedo!", "Voltou?"]);
    }
}
