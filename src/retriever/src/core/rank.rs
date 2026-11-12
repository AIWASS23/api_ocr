//! Ordenação por nota e fusão por posição recíproca (RRF).

use std::collections::HashMap;

use chunking::Document;

use crate::core::documents::{document_key, with_score};

/// `_rank_scores`: maior nota primeiro (empate: ordem original), com a nota no metadado.
pub fn rank_scores(docs: &[Document], scores: &[f64], positive_only: bool, limit: usize) -> Vec<Document> {
    let mut order: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    order
        .into_iter()
        .filter(|(_, s)| !positive_only || *s > 0.0)
        .take(limit)
        .map(|(i, s)| with_score(&docs[i], s))
        .collect()
}

/// RRF ponderado: `Σ peso / (rrf_k + posição)` por documento (deduplicado pela chave), maior
/// primeiro (empate: chave).
pub fn reciprocal_rank_fusion(rankings: &[Vec<Document>], weights: &[f64], rrf_k: f64, k: usize) -> Vec<Document> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut docs: HashMap<String, Document> = HashMap::new();
    for (ranking, w) in rankings.iter().zip(weights) {
        for (rank, doc) in ranking.iter().enumerate() {
            let key = document_key(doc);
            docs.entry(key.clone()).or_insert_with(|| doc.clone());
            *scores.entry(key).or_insert(0.0) += w / (rrf_k + (rank + 1) as f64);
        }
    }
    let mut keys: Vec<String> = scores.keys().cloned().collect();
    keys.sort_by(|a, b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    keys.into_iter().take(k).map(|key| with_score(&docs[&key], scores[&key])).collect()
}
