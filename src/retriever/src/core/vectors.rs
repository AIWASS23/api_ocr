//! Busca densa local: cosseno da consulta com cada documento e MMR (`_local_dense` e
//! `_mmr_indices` do Python, em `float32`).

const EPS: f32 = f32::EPSILON;

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosseno de `query` com cada linha de `matrix`.
pub fn cosine_scores(matrix: &[Vec<f32>], query: &[f32]) -> Vec<f32> {
    let qn = norm(query).max(EPS);
    matrix.iter().map(|row| dot(row, query) / (norm(row) * qn).max(EPS)).collect()
}

/// Índices por nota decrescente (ordenação estável).
pub fn argsort_desc(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// Maximal Marginal Relevance: a cada passo, o candidato com maior
/// `λ·relevância − (1−λ)·máx(similaridade com os já escolhidos)`; empate, o menor índice.
pub fn mmr_indices(matrix: &[Vec<f32>], relevance: &[f32], k: usize, lambda: f32) -> Vec<usize> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let normalized: Vec<Vec<f32>> = matrix.iter().map(|r| { let n = norm(r).max(EPS); r.iter().map(|x| x / n).collect() }).collect();
    let sim = |i: usize, j: usize| dot(&normalized[i], &normalized[j]);
    let first = relevance
        .iter()
        .enumerate()
        .fold(0usize, |best, (i, &v)| if v > relevance[best] { i } else { best });
    let mut selected = vec![first];
    let mut candidates: Vec<usize> = (0..matrix.len()).filter(|&i| i != first).collect();
    while !candidates.is_empty() && selected.len() < k {
        let mut best: Option<(usize, f32)> = None;
        for &i in &candidates {
            let redundancy = selected.iter().map(|&p| sim(i, p)).fold(f32::NEG_INFINITY, f32::max);
            let value = lambda * relevance[i] - (1.0 - lambda) * redundancy;
            if best.is_none_or(|(bi, bv)| value > bv || (value == bv && i < bi)) {
                best = Some((i, value));
            }
        }
        let (chosen, _) = best.expect("há candidatos");
        selected.push(chosen);
        candidates.retain(|&i| i != chosen);
    }
    selected
}
