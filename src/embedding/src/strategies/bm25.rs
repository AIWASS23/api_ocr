//! `bm25` (nova, só lógica) — Okapi BM25 como vetores: o primeiro lote de documentos define o
//! vocabulário, o IDF e o tamanho médio de documento; cada documento vira os pesos BM25 dos seus
//! termos e cada consulta, 1 por termo. O produto escalar consulta·documento é a nota BM25 do
//! documento:
//!
//! `idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`, com
//! `idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5))`.
//!
//! É a perna lexical de uma busca híbrida (junto de um embedding denso), sem modelo. O vetor tem
//! o tamanho do vocabulário (sem limite por padrão): use [`Embedder::embed_sparse`] ou
//! `dimensions`/`max_features` para limitar. Não normalize (`normalize: false`), ou as notas
//! perdem a escala.
//!
//! Opções: `k1` (1.5), `b` (0.75), `query_weight` (`binary` ou `count`) e as do analisador e do
//! vocabulário (`analyzer`, `ngram_range`, `lowercase`, `strip_accents`, `token_pattern`,
//! `stop_words`, `max_df`, `min_df`, `max_features`, `vocabulary`).

use std::sync::{Mutex, PoisonError};

use crate::config::EmbeddingConfig;
use crate::core::options::Options;
use crate::core::vectorizer::{CountVectorizer, SparseRow, Vocabulary, densify};
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

struct Fitted {
    vocab: Vocabulary,
    idf: Vec<f64>,
    avgdl: f64,
}

pub struct Bm25Embedder {
    vectorizer: CountVectorizer,
    k1: f64,
    b: f64,
    binary_query: bool,
    fitted: Mutex<Option<Fitted>>,
}

pub fn load(config: &EmbeddingConfig) -> Result<Bm25Embedder> {
    let mut o = Options::new("bm25", &config.options);
    let k1 = o.f64("k1")?.unwrap_or(1.5);
    let b = o.f64("b")?.unwrap_or(0.75);
    if k1 < 0.0 || !(0.0..=1.0).contains(&b) {
        return Err(EmbeddingError::config("bm25: k1 must be >= 0 and b in [0, 1]"));
    }
    let binary_query = match o.string("query_weight")?.as_deref() {
        None | Some("binary") => true,
        Some("count") => false,
        Some(_) => return Err(EmbeddingError::config("bm25: query_weight must be 'binary' or 'count'")),
    };
    let vectorizer = CountVectorizer::from_options(&mut o, config.dimensions)?;
    o.finish()?;
    Ok(Bm25Embedder { vectorizer, k1, b, binary_query, fitted: Mutex::new(None) })
}

impl Bm25Embedder {
    fn weigh_documents(&self, f: &Fitted, counts: Vec<SparseRow>) -> Vec<SparseRow> {
        counts
            .into_iter()
            .map(|row| {
                let dl: f64 = row.iter().map(|(_, c)| c).sum();
                let norm = self.k1 * (1.0 - self.b + self.b * dl / f.avgdl.max(f64::EPSILON));
                row.into_iter().map(|(i, tf)| (i, f.idf[i] * tf * (self.k1 + 1.0) / (tf + norm))).collect()
            })
            .collect()
    }

    /// Termos do vocabulário aprendido (coluna → termo).
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        self.fitted.lock().unwrap_or_else(PoisonError::into_inner).as_ref().map(|f| f.vocab.terms.clone())
    }
}

impl Embedder for Bm25Embedder {
    fn model_name(&self) -> &str {
        "bm25"
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut fitted = self.fitted.lock().unwrap_or_else(PoisonError::into_inner);
        if fitted.is_none() {
            if input_type == InputType::Query {
                return Err(EmbeddingError::input("BM25 must embed documents before embedding queries"));
            }
            let (vocab, counts) = self.vectorizer.fit_transform(texts)?;
            let n = counts.len() as f64;
            let mut df = vec![0f64; vocab.len()];
            for row in &counts {
                for &(i, _) in row {
                    df[i] += 1.0;
                }
            }
            let idf = df.iter().map(|&d| (1.0 + (n - d + 0.5) / (d + 0.5)).ln()).collect();
            let avgdl = counts.iter().map(|r| r.iter().map(|(_, c)| c).sum::<f64>()).sum::<f64>() / n.max(1.0);
            let f = Fitted { vocab, idf, avgdl };
            let rows = self.weigh_documents(&f, counts);
            let width = f.vocab.len();
            *fitted = Some(f);
            return Ok(rows.iter().map(|r| densify(r, width)).collect());
        }
        let f = fitted.as_ref().ok_or_else(|| EmbeddingError::model("bm25 not fitted"))?;
        let counts = self.vectorizer.transform(&f.vocab, texts);
        let rows = match input_type {
            InputType::Document => self.weigh_documents(f, counts),
            InputType::Query => counts.into_iter().map(|r| r.into_iter().map(|(i, c)| (i, if self.binary_query { 1.0 } else { c })).collect()).collect(),
        };
        Ok(rows.iter().map(|r| densify(r, f.vocab.len())).collect())
    }
}
