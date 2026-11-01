//! Pós-processamento comum (o `_finalize` e o `as_float_array` do Python) e álgebra pequena.

use crate::error::{EmbeddingError, Result};

/// `np.finfo(np.float32).eps`: piso da norma na normalização.
pub const F32_EPS: f32 = f32::EPSILON;

/// Confere a matriz devolvida por um embedder: uma linha por texto, todas com a mesma largura
/// e só valores finitos.
pub fn check(values: &[Vec<f32>], rows: usize) -> Result<()> {
    if values.len() != rows {
        return Err(EmbeddingError::model(format!("Embedder returned {} rows for {rows} input texts", values.len())));
    }
    if let Some(first) = values.first() {
        let width = first.len();
        if values.iter().any(|r| r.len() != width) {
            return Err(EmbeddingError::model("Embeddings must be a 2D matrix (rows of different widths)"));
        }
    }
    if values.iter().flatten().any(|v| !v.is_finite()) {
        return Err(EmbeddingError::model("Embeddings contain NaN or infinite values"));
    }
    Ok(())
}

/// Norma L2 em `f32`, como o `np.linalg.norm` sobre `float32`.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// `v / max(||v||, eps)`.
pub fn normalize_in_place(v: &mut [f32]) {
    let n = l2_norm(v).max(F32_EPS);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// O `_finalize` do `EmbeddingManager`: corta em `dimensions` (Matryoshka) e normaliza.
pub fn finalize(mut values: Vec<Vec<f32>>, rows: usize, dimensions: Option<usize>, normalize: bool) -> Result<Vec<Vec<f32>>> {
    check(&values, rows)?;
    for row in values.iter_mut() {
        if let Some(d) = dimensions
            && row.len() > d
        {
            row.truncate(d);
        }
        if normalize {
            normalize_in_place(row);
        }
    }
    Ok(values)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Similaridade de cosseno (0 quando um dos vetores é nulo).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (na, nb) = (l2_norm(a), l2_norm(b));
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot(a, b) / (na * nb) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_truncates_then_normalizes_and_validates() {
        let out = finalize(vec![vec![3.0, 4.0, 12.0], vec![0.0, 0.0, 1.0]], 2, Some(2), true).unwrap();
        assert_eq!(out[0], vec![0.6, 0.8]);
        assert_eq!(out[1], vec![0.0, 0.0], "vetor nulo continua nulo (piso eps)");
        assert!(finalize(vec![vec![1.0]], 2, None, false).is_err());
        assert!(finalize(vec![vec![1.0], vec![1.0, 2.0]], 2, None, false).is_err());
        assert!(finalize(vec![vec![f32::NAN]], 1, None, false).is_err());
    }
}
