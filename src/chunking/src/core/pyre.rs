//! Expressões regulares com a semântica do módulo `re` do Python.
//!
//! Os separadores do LangChain, do spaCy e do Punkt foram escritos para o Python. Este módulo
//! traduz as diferenças de sintaxe que mudam o resultado e emula `re.split`/`re.finditer`
//! (inclusive matches vazios e grupos de captura), sobre o `fancy-regex` (que tem lookbehind).

use fancy_regex::Regex;

use crate::error::{ChunkError, Result};

/// Regex compilada a partir de um padrão no dialeto do Python.
#[derive(Debug, Clone)]
pub struct PyRegex {
    re: Regex,
    source: String,
}

fn is_rust_meta(c: char) -> bool {
    matches!(c, '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$' | '#' | '&' | '-' | '~' | ' ')
}

/// `{n}`, `{n,}`, `{,m}`, `{n,m}` a partir de `chars[i]` (que é `{`).
fn valid_quantifier(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    let mut digits = 0;
    let mut comma = false;
    while j < chars.len() {
        match chars[j] {
            '0'..='9' => digits += 1,
            ',' if !comma => comma = true,
            '}' => return (digits > 0).then_some(j),
            _ => return None,
        }
        j += 1;
    }
    None
}

/// Traduz um padrão do `re` do Python para a sintaxe do `fancy-regex`.
pub fn translate(pattern: &str) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() + 8);
    let mut in_class = false;
    let mut class_start = 0usize;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() {
            let n = chars[i + 1];
            match n {
                'Z' => out.push_str(r"\z"),
                // Escapes que o Python aceita e o Rust recusa: vira o literal.
                n if n.is_ascii_punctuation() && !is_rust_meta(n) => {
                    if in_class && matches!(n, '[') {
                        out.push_str(r"\[");
                    } else {
                        out.push(n);
                    }
                }
                _ => {
                    out.push('\\');
                    out.push(n);
                }
            }
            i += 2;
            continue;
        }
        if in_class {
            match c {
                ']' if i > class_start => {
                    in_class = false;
                    out.push(']');
                }
                // Dentro de classe, o Python trata `[` como literal (o Rust abriria classe aninhada)
                // e não conhece as operações `&&`, `--`, `~~`.
                '[' => out.push_str(r"\["),
                '&' | '~' if chars.get(i + 1) == Some(&c) => {
                    out.push('\\');
                    out.push(c);
                }
                '-' if chars.get(i + 1) == Some(&'-') => out.push_str(r"\-"),
                _ => out.push(c),
            }
            i += 1;
            continue;
        }
        match c {
            '[' => {
                in_class = true;
                out.push('[');
                // `]` logo após `[` ou `[^` é literal nas duas sintaxes.
                class_start = i + 1;
                if chars.get(i + 1) == Some(&'^') {
                    out.push('^');
                    i += 1;
                    class_start = i + 1;
                }
                if chars.get(class_start) == Some(&']') {
                    out.push_str(r"\]");
                    i = class_start;
                }
            }
            // `$` do Python (sem MULTILINE): fim do texto ou antes de um `\n` final.
            '$' => out.push_str(r"(?=\n?\z)"),
            '{' => match valid_quantifier(&chars, i) {
                Some(end) => {
                    out.extend(&chars[i..=end]);
                    i = end;
                }
                None => out.push_str(r"\{"),
            },
            '}' => out.push_str(r"\}"),
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// Escapa um literal para uso em regex (equivalente de `re.escape`).
pub fn escape(literal: &str) -> String {
    regex::escape(literal)
}

/// Um match: posições em bytes e os grupos de captura (exceto o 0).
#[derive(Debug, Clone)]
pub struct PyMatch {
    pub start: usize,
    pub end: usize,
    pub groups: Vec<Option<(usize, usize)>>,
}

impl PyRegex {
    pub fn new(python_pattern: &str) -> Result<Self> {
        let translated = translate(python_pattern);
        let re = fancy_regex::RegexBuilder::new(&translated)
            .backtrack_limit(10_000_000)
            .build()
            .map_err(|e| ChunkError::Regex(format!("{python_pattern:?}: {e}")))?;
        Ok(Self { re, source: python_pattern.to_string() })
    }

    /// Para literais: nenhum metacaractere.
    pub fn literal(text: &str) -> Result<Self> {
        Self::new(&escape(text))
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    fn find_at(&self, text: &str, pos: usize) -> Result<Option<PyMatch>> {
        let caps = self
            .re
            .captures_from_pos(text, pos)
            .map_err(|e| ChunkError::Regex(format!("{:?}: {e}", self.source)))?;
        Ok(caps.map(|c| {
            let m = c.get(0).expect("grupo 0 sempre existe");
            PyMatch {
                start: m.start(),
                end: m.end(),
                groups: (1..c.len()).map(|g| c.get(g).map(|m| (m.start(), m.end()))).collect(),
            }
        }))
    }

    /// `re.search(p, text) is not None`.
    pub fn is_match(&self, text: &str) -> Result<bool> {
        self.re.is_match(text).map_err(|e| ChunkError::Regex(format!("{:?}: {e}", self.source)))
    }

    /// `re.search` a partir do início.
    pub fn search(&self, text: &str) -> Result<Option<PyMatch>> {
        self.find_at(text, 0)
    }

    /// `re.match` (ancorado no início).
    pub fn match_start(&self, text: &str) -> Result<Option<PyMatch>> {
        Ok(self.find_at(text, 0)?.filter(|m| m.start == 0))
    }

    /// `re.finditer`: depois de um match vazio, o próximo precisa avançar.
    pub fn find_iter(&self, text: &str) -> Result<Vec<PyMatch>> {
        let mut out = Vec::new();
        let mut pos = 0;
        let mut must_advance = false;
        while pos <= text.len() {
            let Some(mut m) = self.find_at(text, pos)? else { break };
            if must_advance && m.start == pos && m.end == pos {
                // Proibido repetir o match vazio na mesma posição: tenta a partir do próximo char.
                let Some(next) = text[pos..].chars().next() else { break };
                match self.find_at(text, pos + next.len_utf8())? {
                    Some(n) => m = n,
                    None => break,
                }
            }
            must_advance = m.start == m.end;
            pos = m.end;
            out.push(m);
        }
        Ok(out)
    }

    /// `re.split(f"({pattern})", text)` quando `capture` é verdadeiro; `re.split(pattern, text)`
    /// caso contrário. Grupos internos do padrão também entram na lista, como no Python
    /// (um grupo que não participou vira texto vazio).
    pub fn split<'t>(&self, text: &'t str, capture: bool) -> Result<Vec<&'t str>> {
        let mut out = Vec::new();
        let mut last = 0;
        for m in self.find_iter(text)? {
            out.push(&text[last..m.start]);
            if capture {
                out.push(&text[m.start..m.end]);
            }
            for g in &m.groups {
                out.push(g.map_or("", |(s, e)| &text[s..e]));
            }
            last = m.end;
        }
        out.push(&text[last..]);
        Ok(out)
    }

    /// `re.sub(pattern, repl, text)` com substituição literal.
    pub fn replace_all(&self, text: &str, repl: &str) -> Result<String> {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for m in self.find_iter(text)? {
            out.push_str(&text[last..m.start]);
            out.push_str(repl);
            last = m.end;
        }
        out.push_str(&text[last..]);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dollar_matches_before_final_newline() {
        let r = PyRegex::new("$").unwrap();
        assert_eq!(r.find_iter("ab\n").unwrap().iter().map(|m| m.start).collect::<Vec<_>>(), [2, 3]);
    }

    #[test]
    fn python_literals_translate() {
        for p in ["\\\\chapter{", "[\\[\\]]", "a{2}", "x{,3}", "[--]", "\\`", "[^\\(\\\"\\`{\\[:;&\\#\\*@\\)}\\]\\-,]"] {
            PyRegex::new(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        }
        assert!(PyRegex::new("a{2}").unwrap().is_match("aa").unwrap());
        assert!(PyRegex::new("\\\\chapter{").unwrap().is_match("\\chapter{x}").unwrap());
    }

    #[test]
    fn split_like_python() {
        let r = PyRegex::new(r"(?<=[.!?])\s+").unwrap();
        assert_eq!(r.split("A. B! C", true).unwrap(), ["A.", " ", "B!", " ", "C"]);
        assert_eq!(r.split("A. B", false).unwrap(), ["A.", "B"]);
        // Grupo interno entra na lista, como no Python.
        let g = PyRegex::new(r"(-)").unwrap();
        assert_eq!(g.split("a-b", false).unwrap(), ["a", "-", "b"]);
    }
}
