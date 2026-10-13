//! Semântica de strings do Python 3, para que os cortes saiam iguais aos do chatbot.
//!
//! O Python mede texto em *code points* (`len`), considera espaço um conjunto um pouco maior
//! que o `White_Space` do Unicode (inclui `\x1c`–`\x1f`) e tem regras próprias para
//! `isprintable`, `splitlines` e `strip`. Tudo que depende disso passa por aqui.

use unicode_general_category::{GeneralCategory, get_general_category};

/// `str.isspace()` para um caractere.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.isspace()`: não vazio e só espaços.
pub fn is_all_space(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_space)
}

/// `str.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_space)
}

pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `len(s)` do Python.
pub fn len(s: &str) -> usize {
    s.chars().count()
}

/// `str.isprintable()` para um caractere: falso para Cc, Cf, Cs, Co, Cn, Zl, Zp e Zs (exceto o
/// espaço ASCII). Remove, por exemplo, o espaço inquebrável e o hífen invisível.
pub fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    !matches!(
        get_general_category(c),
        GeneralCategory::Control
            | GeneralCategory::Format
            | GeneralCategory::Surrogate
            | GeneralCategory::PrivateUse
            | GeneralCategory::Unassigned
            | GeneralCategory::LineSeparator
            | GeneralCategory::ParagraphSeparator
            | GeneralCategory::SpaceSeparator
    )
}

/// Categoria Unicode começa com "P" (pontuação) — `unicodedata.category(c).startswith("P")`.
pub fn is_punctuation(c: char) -> bool {
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

/// Quebras de linha reconhecidas por `str.splitlines()`.
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// `str.splitlines(keepends=...)`; `\r\n` conta como uma quebra só.
pub fn splitlines(s: &str, keepends: bool) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut iter = s.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        if is_line_break(c) {
            let mut end = i + c.len_utf8();
            if c == '\r'
                && let Some(&(j, '\n')) = iter.peek()
            {
                iter.next();
                end = j + 1;
            }
            out.push(if keepends { &s[start..end] } else { &s[start..i] });
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `str.split()` sem argumentos (sequências de espaço, sem vazios).
pub fn split_whitespace(s: &str) -> Vec<&str> {
    s.split(is_space).filter(|p| !p.is_empty()).collect()
}

/// `str.isupper()` / `str.islower()` para o primeiro caractere.
pub fn first_is_upper(s: &str) -> bool {
    s.chars().next().is_some_and(char::is_uppercase)
}

pub fn first_is_lower(s: &str) -> bool {
    s.chars().next().is_some_and(char::is_lowercase)
}

/// Índice de byte do n-ésimo caractere (ou o fim).
pub fn byte_at_char(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

/// Equivalente de `utils/output.py::normalize_chunks`: `strip()` e descarta vazios.
pub fn normalize_chunks<I, S>(chunks: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    chunks
        .into_iter()
        .filter_map(|c| {
            let v = strip(c.as_ref());
            (!v.is_empty()).then(|| v.to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_whitespace_and_printable() {
        assert!(is_space('\u{1c}') && !'\u{1c}'.is_whitespace());
        assert!(!is_printable('\u{a0}'), "NBSP não é imprimível no Python");
        assert!(!is_printable('\u{ad}'), "soft hyphen (Cf)");
        assert!(is_printable('ç') && is_printable(' '));
        assert_eq!(strip("\u{1c} a \u{a0}"), "a");
    }

    #[test]
    fn splitlines_like_python() {
        assert_eq!(splitlines("a\r\nb\rc\u{2028}d\n", true), ["a\r\n", "b\r", "c\u{2028}", "d\n"]);
        assert_eq!(splitlines("a\n\nb", false), ["a", "", "b"]);
        assert!(splitlines("", true).is_empty());
    }

    #[test]
    fn normalize_drops_blank() {
        assert_eq!(normalize_chunks([" a ", "\n", "", "b"]), ["a", "b"]);
    }
}
