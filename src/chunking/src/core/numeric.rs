//! O pedaço do NumPy que o `SemanticChunker` usa, com a mesma aritmética: soma em pares
//! (`pairwise_sum`), `percentile` linear com o `_lerp` do NumPy, `std` populacional e
//! `gradient` com espaçamento uniforme.

/// `np.add.reduce` para float64 (soma em pares, blocos de 8, como no NumPy).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        // O NumPy começa de -0.0 e soma em ordem.
        return a.iter().fold(-0.0, |acc, x| acc + x);
    }
    if n <= BLOCK {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

pub fn mean(a: &[f64]) -> f64 {
    pairwise_sum(a) / a.len() as f64
}

/// `np.std` (ddof=0).
pub fn std(a: &[f64]) -> f64 {
    let m = mean(a);
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (pairwise_sum(&sq) / a.len() as f64).sqrt()
}

/// `np.percentile(a, q)` com o método padrão (linear).
pub fn percentile(a: &[f64], q: f64) -> f64 {
    let mut s = a.to_vec();
    s.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    if n == 0 {
        return f64::NAN;
    }
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let prev = virtual_index.floor().clamp(0.0, (n - 1) as f64);
    let gamma = virtual_index - prev;
    let lo = prev as usize;
    let hi = (lo + 1).min(n - 1);
    let (a0, b0) = (s[lo], s[hi]);
    // `_lerp` do NumPy: exato nas pontas.
    let diff = b0 - a0;
    if gamma >= 0.5 { b0 - diff * (1.0 - gamma) } else { a0 + diff * gamma }
}

/// `np.gradient(a, range(len(a)))` (espaçamento 1, bordas de primeira ordem).
pub fn gradient(a: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n < 2 {
        return a.iter().map(|_| 0.0).collect();
    }
    let mut g = vec![0.0; n];
    g[0] = a[1] - a[0];
    g[n - 1] = a[n - 1] - a[n - 2];
    for i in 1..n - 1 {
        g[i] = (a[i + 1] - a[i - 1]) / 2.0;
    }
    g
}

/// Similaridade de cosseno com NaN/inf trocados por 0, como em `langchain_core.utils.math`.
pub fn cosine_similarity(x: &[f32], y: &[f32]) -> f64 {
    let dot: f64 = x.iter().zip(y).map(|(a, b)| *a as f64 * *b as f64).sum();
    let nx: f64 = x.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let ny: f64 = y.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let s = dot / (nx * ny);
    if s.is_finite() { s } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy() {
        let d = [0.1, 0.5, 0.2, 0.9, 0.3];
        assert_eq!(percentile(&d, 95.0), 0.82);
        assert!((percentile(&d, 25.0) - 0.2).abs() < 1e-12);
        assert_eq!(std(&d), 0.282842712474619);
        assert_eq!(gradient(&[1.0, 2.0, 4.0, 7.0]), [1.0, 1.5, 2.5, 3.0]);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
