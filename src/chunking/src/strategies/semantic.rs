//! `semantic` — port do `SemanticChunker` (langchain_experimental): divide em frases, gera o
//! embedding de cada frase junto com as vizinhas (`buffer_size`), mede a distância de cosseno
//! entre frases consecutivas e corta onde a distância passa do limiar
//! (`percentile`, `standard_deviation`, `interquartile` ou `gradient`). Depois, como no
//! `ChunkSemantic` do chatbot, impõe o tamanho com o splitter recursivo (`keep_separator="end"`).
//! Se os embeddings falharem, cai para o corte recursivo (mesmo comportamento do Python).
//!
//! Opções: `buffer_size` (1), `breakpoint_threshold_amount`, `number_of_chunks`,
//! `sentence_split_regex` (`(?<=[.?!])\s+`), `min_chunk_size`, `add_start_index` (ignorada).

use std::sync::Arc;

use crate::chunker::{Chunker, Sizing};
use crate::core::numeric::{cosine_similarity, gradient, mean, percentile, std};
use crate::core::options::Options;
use crate::core::pyre::PyRegex;
use crate::core::pystr::{self, normalize_chunks};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, TextSplitter};
use crate::error::{ChunkError, Result};
use crate::models::embeddings::Embedder;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Breakpoint {
    Percentile,
    StandardDeviation,
    Interquartile,
    Gradient,
}

impl Breakpoint {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "percentile" => Breakpoint::Percentile,
            "standard_deviation" => Breakpoint::StandardDeviation,
            "interquartile" => Breakpoint::Interquartile,
            "gradient" => Breakpoint::Gradient,
            other => return Err(ChunkError::config(format!("Got unexpected `breakpoint_threshold_type`: {other}"))),
        })
    }

    fn default_amount(self) -> f64 {
        match self {
            Breakpoint::Percentile | Breakpoint::Gradient => 95.0,
            Breakpoint::StandardDeviation => 3.0,
            Breakpoint::Interquartile => 1.5,
        }
    }
}

pub struct ChunkSemantic {
    sizing: Sizing,
    embedder: Arc<dyn Embedder>,
    breakpoint: Breakpoint,
    amount: f64,
    buffer_size: usize,
    number_of_chunks: Option<usize>,
    min_chunk_size: Option<usize>,
    sentence_split: PyRegex,
    size_splitter: RecursiveSplitter,
}

impl ChunkSemantic {
    pub fn new(embedder: Arc<dyn Embedder>, sizing: Sizing, breakpoint: &str, mut o: Options) -> Result<Self> {
        let breakpoint = Breakpoint::parse(breakpoint)?;
        let buffer_size = o.usize("buffer_size")?.unwrap_or(1);
        let amount = o.f64("breakpoint_threshold_amount")?.unwrap_or(breakpoint.default_amount());
        let number_of_chunks = o.usize("number_of_chunks")?;
        let min_chunk_size = o.usize("min_chunk_size")?;
        let regex = o.string("sentence_split_regex")?.unwrap_or_else(|| r"(?<=[.?!])\s+".to_string());
        o.ignore(&["add_start_index"]);
        o.finish()?;
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_keep_separator(KeepSeparator::End);
        Ok(Self {
            sizing,
            embedder,
            breakpoint,
            amount,
            buffer_size,
            number_of_chunks,
            min_chunk_size,
            sentence_split: PyRegex::new(&regex)?,
            size_splitter: RecursiveSplitter::new(base, &[] as &[&str], false)?,
        })
    }

    /// `combine_sentences`: cada frase com `buffer_size` vizinhas de cada lado.
    fn combine(&self, sentences: &[&str]) -> Vec<String> {
        let b = self.buffer_size;
        (0..sentences.len())
            .map(|i| {
                let mut s = String::new();
                for before in &sentences[i.saturating_sub(b)..i] {
                    s.push_str(before);
                    s.push(' ');
                }
                s.push_str(sentences[i]);
                for after in &sentences[i + 1..(i + 1 + b).min(sentences.len())] {
                    s.push(' ');
                    s.push_str(after);
                }
                s
            })
            .collect()
    }

    fn threshold(&self, distances: &[f64]) -> (f64, Vec<f64>) {
        match self.breakpoint {
            Breakpoint::Percentile => (percentile(distances, self.amount), distances.to_vec()),
            Breakpoint::StandardDeviation => (mean(distances) + self.amount * std(distances), distances.to_vec()),
            Breakpoint::Interquartile => {
                let iqr = percentile(distances, 75.0) - percentile(distances, 25.0);
                (mean(distances) + self.amount * iqr, distances.to_vec())
            }
            Breakpoint::Gradient => {
                let g = gradient(distances);
                (percentile(&g, self.amount), g)
            }
        }
    }

    /// `_threshold_from_clusters` (quando `number_of_chunks` é dado).
    fn threshold_from_clusters(&self, distances: &[f64], n: usize) -> f64 {
        let (x1, y1) = (distances.len() as f64, 0.0);
        let (x2, y2) = (1.0, 100.0);
        let x = (n as f64).min(x1).max(x2);
        let y = if x2 == x1 { y2 } else { y1 + ((y2 - y1) / (x2 - x1)) * (x - x1) };
        percentile(distances, y.clamp(0.0, 100.0))
    }

    /// `SemanticChunker.split_text`.
    pub fn semantic_split(&self, text: &str) -> Result<Vec<String>> {
        let sentences = self.sentence_split.split(text, false)?;
        if sentences.len() == 1 {
            return Ok(vec![sentences[0].to_string()]);
        }
        if self.breakpoint == Breakpoint::Gradient && sentences.len() == 2 {
            return Ok(sentences.iter().map(|s| s.to_string()).collect());
        }
        let combined = self.combine(&sentences);
        let embeddings = self.embedder.embed_documents(&combined)?;
        if embeddings.len() != sentences.len() {
            return Err(ChunkError::Model("o modelo devolveu um número de vetores diferente do de frases".into()));
        }
        let distances: Vec<f64> = embeddings.windows(2).map(|w| 1.0 - cosine_similarity(&w[0], &w[1])).collect();
        let (threshold, array) = match self.number_of_chunks {
            Some(n) => (self.threshold_from_clusters(&distances, n), distances.clone()),
            None => self.threshold(&distances),
        };
        let mut chunks = Vec::new();
        let mut start = 0;
        for (index, x) in array.iter().enumerate() {
            if *x <= threshold {
                continue;
            }
            let combined = sentences[start..=index].join(" ");
            if self.min_chunk_size.is_some_and(|m| pystr::len(&combined) < m) {
                continue;
            }
            chunks.push(combined);
            start = index + 1;
        }
        if start < sentences.len() {
            chunks.push(sentences[start..].join(" "));
        }
        Ok(chunks)
    }

    fn fallback(&self, text: &str) -> Result<Vec<String>> {
        Ok(normalize_chunks(self.size_splitter.split_text(text)?))
    }
}

impl Chunker for ChunkSemantic {
    crate::sizing_accessors!("semantic");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if pystr::strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let semantic = match self.semantic_split(text) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[WARN] Erro no chunking semântico: {e}. Usando chunking recursivo.");
                return self.fallback(text);
            }
        };
        let chunks: Vec<String> = semantic.into_iter().filter(|c| !pystr::strip(c).is_empty()).collect();
        if chunks.is_empty() {
            return self.fallback(text);
        }
        let mut sized = Vec::new();
        for c in &chunks {
            sized.extend(self.size_splitter.split_text(c)?);
        }
        let out = normalize_chunks(sized);
        Ok(if out.is_empty() { normalize_chunks(chunks) } else { out })
    }
}
