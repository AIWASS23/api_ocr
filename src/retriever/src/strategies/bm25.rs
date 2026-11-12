//! `bm25` — Okapi BM25 sobre o acervo (documentos ou `provider.documents`): termos `\w+` em
//! minúsculas; `idf = ln(1 + (N − df + 0,5)/(df + 0,5))`; nota = Σ idf·(tf·(k1+1)/(tf +
//! k1·(1 − b + b·|d|/avgdl)) + δ). Opções: `k1` (1,5), `b` (0,75), `delta` (0; 1 no `bm25_plus`).

use std::collections::{HashMap, HashSet};

use chunking::Document;

use crate::context::Ctx;
use crate::core::text::tokens;
use crate::error::Result;

/// BM25 (ou BM25+, com `plus`) de todo o acervo; documentos sem termo da consulta ficam no fim.
pub fn score(ctx: &Ctx, query: &str, plus: bool) -> Result<Vec<Document>> {
    let docs = ctx.corpus()?;
    let tokenized: Vec<Vec<String>> = docs.iter().map(|d| tokens(&d.page_content)).collect();
    if tokenized.is_empty() {
        return Ok(Vec::new());
    }
    let q = tokens(query);
    let mut df: HashMap<&str, f64> = HashMap::new();
    for terms in &tokenized {
        for t in terms.iter().map(String::as_str).collect::<HashSet<_>>() {
            *df.entry(t).or_insert(0.0) += 1.0;
        }
    }
    let n = tokenized.len() as f64;
    let avg = tokenized.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let avg = if avg == 0.0 { 1.0 } else { avg };
    let k1 = ctx.cfg.option_f64("k1", 1.5);
    let b = ctx.cfg.option_f64("b", 0.75);
    let delta = ctx.cfg.option_f64("delta", if plus { 1.0 } else { 0.0 });
    let scores: Vec<f64> = tokenized
        .iter()
        .map(|terms| {
            let mut tf: HashMap<&str, f64> = HashMap::new();
            for t in terms {
                *tf.entry(t.as_str()).or_insert(0.0) += 1.0;
            }
            q.iter()
                .filter_map(|t| {
                    let f = *tf.get(t.as_str())?;
                    let d = df[t.as_str()];
                    let idf = (1.0 + (n - d + 0.5) / (d + 0.5)).ln();
                    let denom = f + k1 * (1.0 - b + b * terms.len() as f64 / avg);
                    Some(idf * (f * (k1 + 1.0) / denom + delta))
                })
                .sum()
        })
        .collect();
    Ok(ctx.rank(&docs, &scores, false, None))
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    score(ctx, query, false)
}
