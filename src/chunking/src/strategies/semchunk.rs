//! `semchunk` — port do algoritmo da biblioteca `semchunk` (4.x): divide primeiro pelos
//! separadores mais significativos que existirem no texto (sequências de quebra de linha,
//! tabs, espaços, depois pontuação de frase/oração...), recursivamente, e junta os pedaços com
//! busca binária até o limite de tokens. A sobreposição é a do wrapper `ChunkSemchunk` do
//! chatbot: o fim (em tokens) do chunk anterior é prefixado ao seguinte.
//!
//! Precisa de tokenizador (`manager.load_model("semchunk")` ou injeção).

use std::sync::Arc;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, normalize_chunks};
use crate::error::Result;
use crate::models::tokenizer::TextTokenizer;

/// Separadores não-espaço, do mais ao menos desejável (`_NON_WHITESPACE_STRUCTURAL_SPLITTERS`).
const NON_WHITESPACE_SPLITTERS: [&str; 25] = [
    ".", "?", "!", ";", ",", "(", ")", "[", "]", "“", "”", "‘", "’", "'", "\"", "`", "*", ":", "—", "…", "/", "\\", "–", "&",
    "-",
];

/// O maior trecho de caracteres que satisfazem `pred` com tamanho mínimo `min` (o primeiro, em
/// caso de empate, como `max(runs, key=len)`).
fn longest_run(text: &str, pred: impl Fn(char) -> bool, min: usize) -> Option<&str> {
    let mut best: Option<(usize, usize, usize)> = None; // start, end, len
    let mut cur: Option<(usize, usize)> = None; // start, len
    for (i, c) in text.char_indices() {
        if pred(c) {
            cur = Some(cur.map_or((i, 1), |(s, l)| (s, l + 1)));
        } else if let Some((s, l)) = cur.take()
            && l >= min
            && best.is_none_or(|b| l > b.2)
        {
            best = Some((s, i, l));
        }
    }
    if let Some((s, l)) = cur
        && l >= min
        && best.is_none_or(|b| l > b.2)
    {
        best = Some((s, text.len(), l));
    }
    best.map(|(s, e, _)| &text[s..e])
}

/// `_split_text`: (separador, separador é espaço?, pedaços).
fn split_text(text: &str) -> (String, bool, Vec<&str>) {
    let is_nl = |c: char| c == '\n' || c == '\r';
    if text.contains(['\n', '\r']) {
        let sep = longest_run(text, is_nl, 2)
            .map(str::to_string)
            .unwrap_or_else(|| text.chars().find(|c| is_nl(*c)).expect("tem quebra").to_string());
        return (sep.clone(), true, text.split(sep.as_str()).collect());
    }
    if text.contains('\t') {
        let sep = longest_run(text, |c| c == '\t', 2).unwrap_or("\t").to_string();
        return (sep.clone(), true, text.split(sep.as_str()).collect());
    }
    if let Some(ws) = text.chars().find(|c| pystr::is_space(*c)) {
        if let Some(run) = longest_run(text, pystr::is_space, 2) {
            let sep = run.to_string();
            return (sep.clone(), true, text.split(sep.as_str()).collect());
        }
        // Espaço precedido de pontuação estrutural tem prioridade.
        for preceder in NON_WHITESPACE_SPLITTERS {
            let mut found: Option<char> = None;
            for (i, _) in text.match_indices(preceder) {
                if let Some(c) = text[i + preceder.len()..].chars().next()
                    && pystr::is_space(c)
                {
                    found = Some(c);
                    break;
                }
            }
            if let Some(sp) = found {
                // `re.split(rf"(?<={preceder}){splitter}", text)`.
                let pat = format!("{preceder}{sp}");
                let mut parts = Vec::new();
                let mut last = 0;
                let mut from = 0;
                while let Some(off) = text[from..].find(&pat) {
                    let at = from + off + preceder.len();
                    parts.push(&text[last..at]);
                    last = at + sp.len_utf8();
                    from = last;
                }
                parts.push(&text[last..]);
                return (sp.to_string(), true, parts);
            }
        }
        let sep = ws.to_string();
        return (sep.clone(), true, text.split(sep.as_str()).collect());
    }
    for s in NON_WHITESPACE_SPLITTERS {
        if text.contains(s) {
            return (s.to_string(), false, text.split(s).collect());
        }
    }
    let chars: Vec<&str> = text.char_indices().map(|(i, c)| &text[i..i + c.len_utf8()]).collect();
    (String::new(), true, chars)
}

struct Semchunk<'a> {
    count: &'a dyn Fn(&str) -> usize,
}

impl Semchunk<'_> {
    /// `merge_splits`: busca binária do maior prefixo de pedaços que cabe em `chunk_size`.
    #[allow(clippy::too_many_arguments)]
    fn merge(
        &self,
        text: &str,
        char_starts: &[usize],
        byte_starts: &[usize],
        sep_chars: usize,
        sep_bytes: usize,
        chunk_size: usize,
        start: usize,
        mut high: usize,
    ) -> (usize, String) {
        let slice = |a: usize, b: usize| -> &str {
            let (s, e) = (byte_starts[a], byte_starts[b].saturating_sub(sep_bytes));
            if s < e { &text[s..e] } else { "" }
        };
        let mut average = 0.2f64;
        let mut low = start;
        let offset = char_starts[start];
        let mut target = offset as f64 + chunk_size as f64 * average;
        while low < high {
            let i = low + char_starts[low..high].partition_point(|&c| (c as f64) < target);
            let midpoint = i.min(high - 1);
            let tokens = (self.count)(slice(start, midpoint));
            let local_cum = char_starts[midpoint] - offset;
            if local_cum > 0 && tokens > 0 {
                average = local_cum as f64 / tokens as f64;
                target = offset as f64 + chunk_size as f64 * average;
            }
            if tokens > chunk_size {
                high = midpoint;
            } else {
                low = midpoint + 1;
            }
        }
        let end = low - 1;
        let _ = sep_chars;
        (end, slice(start, end).to_string())
    }

    /// `chunk()` (sem sobreposição nem offsets).
    fn chunk(&self, text: &str, chunk_size: usize, first: bool) -> Vec<String> {
        let (splitter, is_ws, splits) = split_text(text);
        let sep_chars = pystr::len(&splitter);
        let sep_bytes = splitter.len();
        let mut char_starts = Vec::with_capacity(splits.len() + 1);
        let mut byte_starts = Vec::with_capacity(splits.len() + 1);
        let (mut c, mut b) = (0usize, 0usize);
        char_starts.push(0);
        byte_starts.push(0);
        for s in &splits {
            c += pystr::len(s) + sep_chars;
            b += s.len() + sep_bytes;
            char_starts.push(c);
            byte_starts.push(b);
        }
        let n = splits.len();
        let mut chunks: Vec<String> = Vec::new();
        let mut i = 0;
        while i < n {
            let split = splits[i];
            let next_i = if (self.count)(split) > chunk_size {
                chunks.extend(self.chunk(split, chunk_size, false));
                i + 1
            } else {
                let (end, merged) =
                    self.merge(text, &char_starts, &byte_starts, sep_chars, sep_bytes, chunk_size, i, n + 1);
                chunks.push(merged);
                end
            };
            if !is_ws && next_i < n {
                let with_sep = format!("{}{}", chunks.last().map_or("", String::as_str), splitter);
                if (self.count)(&with_sep) <= chunk_size {
                    if let Some(last) = chunks.last_mut() {
                        *last = with_sep;
                    }
                } else {
                    chunks.push(splitter.clone());
                }
            }
            i = next_i;
        }
        if first {
            chunks.retain(|c| !c.is_empty() && !pystr::is_all_space(c));
        }
        chunks
    }
}

/// `semchunk.chunk(text, chunk_size, token_counter)` (a função da biblioteca, sem sobreposição).
pub fn semchunk(text: &str, chunk_size: usize, count: &dyn Fn(&str) -> usize) -> Vec<String> {
    Semchunk { count }.chunk(text, chunk_size, true)
}

pub struct ChunkSemchunk {
    sizing: Sizing,
    tokenizer: Arc<dyn TextTokenizer>,
}

impl ChunkSemchunk {
    /// Opções aceitas e ignoradas (do `Chunker.__call__` do semchunk): `processes`, `progress`.
    pub fn new(tokenizer: Arc<dyn TextTokenizer>, sizing: Sizing, mut o: Options) -> Result<Self> {
        o.ignore(&["processes", "progress"]);
        o.finish()?;
        Ok(Self { sizing, tokenizer })
    }

    fn count(&self, s: &str) -> usize {
        self.tokenizer.count(s)
    }

    /// `_overlap_text`: o maior sufixo (em palavras `\S+\s*`) com até `chunk_overlap` tokens.
    fn overlap_text(&self, text: &str) -> String {
        if self.sizing.chunk_overlap == 0 {
            return String::new();
        }
        let mut pieces: Vec<&str> = Vec::new();
        let mut start: Option<usize> = None;
        let mut in_ws = false;
        for (i, c) in text.char_indices() {
            let ws = pystr::is_space(c);
            match (start, ws) {
                (None, false) => {
                    start = Some(i);
                    in_ws = false;
                }
                (Some(s), false) if in_ws => {
                    pieces.push(&text[s..i]);
                    start = Some(i);
                    in_ws = false;
                }
                (Some(_), true) => in_ws = true,
                _ => {}
            }
        }
        if let Some(s) = start {
            pieces.push(&text[s..]);
        }
        let mut suffix = String::new();
        for piece in pieces.iter().rev() {
            let candidate = format!("{piece}{suffix}");
            if self.count(&candidate) > self.sizing.chunk_overlap {
                break;
            }
            suffix = candidate;
        }
        pystr::strip(&suffix).to_string()
    }

    /// `_merge_overlap`: prefixa o anterior e corta palavras do prefixo até caber.
    fn merge_overlap(&self, mut prefix: String, current: &str) -> String {
        let join = |p: &str| if p.is_empty() { current.to_string() } else { format!("{p} {current}") };
        let mut candidate = join(&prefix);
        while !prefix.is_empty() && self.count(&candidate) > self.sizing.chunk_size {
            prefix = if prefix.contains(' ') {
                // `prefix.split(maxsplit=1)[1]`
                let t = pystr::lstrip(&prefix);
                let cut = t.find(pystr::is_space).unwrap_or(t.len());
                pystr::lstrip(&t[cut..]).to_string()
            } else {
                String::new()
            };
            candidate = join(&prefix);
        }
        candidate
    }
}

impl Chunker for ChunkSemchunk {
    crate::sizing_accessors!("semchunk");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if pystr::strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let content = self.sizing.chunk_size - self.sizing.chunk_overlap;
        let count = |s: &str| self.count(s);
        let chunks = semchunk(text, content, &count);
        if self.sizing.chunk_overlap == 0 || chunks.len() < 2 {
            return Ok(normalize_chunks(chunks));
        }
        let mut out = vec![chunks[0].clone()];
        for w in chunks.windows(2) {
            let prefix = self.overlap_text(&w[0]);
            out.push(self.merge_overlap(prefix, &w[1]));
        }
        Ok(normalize_chunks(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_the_most_significant_separator() {
        let (sep, ws, parts) = split_text("a\n\nb\nc");
        assert_eq!((sep.as_str(), ws), ("\n\n", true));
        assert_eq!(parts, ["a", "b\nc"]);
        let (sep, _, parts) = split_text("um. dois três");
        assert_eq!(sep, " ");
        assert_eq!(parts, ["um.", "dois três"]);
    }

    #[test]
    fn chunks_respect_the_budget() {
        let words = |s: &str| s.split_whitespace().count();
        let text = "Um dois três quatro. Cinco seis sete oito. Nove dez onze doze.";
        let chunks = semchunk(text, 4, &words);
        assert!(chunks.iter().all(|c| words(c) <= 4), "{chunks:?}");
        assert_eq!(chunks.join(" ").split_whitespace().count(), 12);
    }
}
