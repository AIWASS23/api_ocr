//! SVD truncado aleatorizado (Halko, Martinsson e Tropp, 2011) — o algoritmo padrão do
//! `TruncatedSVD` do scikit-learn — sobre uma matriz esparsa por linhas. Determinístico: a
//! matriz aleatória vem de uma semente.

use crate::core::vectorizer::SparseRow;

/// Gerador xorshift64* com normal por Box-Muller.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.next_f64(), self.next_f64());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// `Y = X · M` (n×l), com `M` dado por colunas (l colunas de tamanho m).
fn mul(x: &[SparseRow], m_cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; x.len()]; m_cols.len()];
    for (j, col) in m_cols.iter().enumerate() {
        for (r, row) in x.iter().enumerate() {
            out[j][r] = row.iter().map(|&(i, v)| v * col[i]).sum();
        }
    }
    out
}

/// `Z = Xᵀ · Y` (m×l), com `Y` dado por colunas (l colunas de tamanho n).
fn mul_t(x: &[SparseRow], m: usize, y_cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; m]; y_cols.len()];
    for (j, col) in y_cols.iter().enumerate() {
        for (r, row) in x.iter().enumerate() {
            let c = col[r];
            if c != 0.0 {
                for &(i, v) in row {
                    out[j][i] += v * c;
                }
            }
        }
    }
    out
}

fn dotv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Gram-Schmidt modificado com reortogonalização; colunas dependentes viram zero.
fn orthonormalize(cols: &mut [Vec<f64>]) {
    for j in 0..cols.len() {
        for _ in 0..2 {
            for k in 0..j {
                let p = dotv(&cols[k], &cols[j]);
                let (a, b) = cols.split_at_mut(j);
                for (x, y) in b[0].iter_mut().zip(&a[k]) {
                    *x -= p * y;
                }
            }
        }
        let n = dotv(&cols[j], &cols[j]).sqrt();
        if n > 1e-12 {
            cols[j].iter_mut().for_each(|x| *x /= n);
        } else {
            cols[j].iter_mut().for_each(|x| *x = 0.0);
        }
    }
}

/// Autovalores/autovetores de uma matriz simétrica pequena (Jacobi cíclico), em ordem
/// decrescente de autovalor. Autovetores por coluna: `vecs[k]` é o k-ésimo.
fn jacobi_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        let total: f64 = a.iter().flatten().map(|x| x * x).sum();
        if off <= 1e-24 * total.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (lo, hi) = a.split_at_mut(q);
                for (x, y) in lo[p].iter_mut().zip(hi[0].iter_mut()) {
                    let (apk, aqk) = (*x, *y);
                    *x = c * apk - s * aqk;
                    *y = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[y][y].partial_cmp(&a[x][x]).unwrap_or(std::cmp::Ordering::Equal));
    let values = order.iter().map(|&k| a[k][k]).collect();
    let vectors = order.iter().map(|&k| v.iter().map(|row| row[k]).collect()).collect();
    (values, vectors)
}

/// Componentes do SVD truncado: `k` vetores singulares à direita (cada um com `m` posições),
/// os da maior variância primeiro, com o sinal da maior coordenada positivo (`svd_flip`).
pub fn truncated_svd(x: &[SparseRow], m: usize, k: usize, n_oversamples: usize, n_iter: usize, seed: u64) -> Vec<Vec<f64>> {
    let k = k.min(x.len()).min(m);
    if k == 0 {
        return Vec::new();
    }
    let l = (k + n_oversamples).min(m).min(x.len()).max(k);
    let mut rng = Rng::new(seed);
    let omega: Vec<Vec<f64>> = (0..l).map(|_| (0..m).map(|_| rng.normal()).collect()).collect();
    let mut q = mul(x, &omega);
    for _ in 0..n_iter {
        orthonormalize(&mut q);
        let mut z = mul_t(x, m, &q);
        orthonormalize(&mut z);
        q = mul(x, &z);
    }
    orthonormalize(&mut q);
    // B = Qᵀ X (l×m), guardada por linhas = colunas de Xᵀ Q.
    let b = mul_t(x, m, &q);
    let gram: Vec<Vec<f64>> = (0..l).map(|i| (0..l).map(|j| dotv(&b[i], &b[j])).collect()).collect();
    let (values, vectors) = jacobi_eigen(gram);
    let mut components = Vec::with_capacity(k);
    for (idx, u) in vectors.iter().enumerate().take(k) {
        let sigma = values[idx].max(0.0).sqrt();
        let mut v = vec![0.0; m];
        if sigma > 1e-12 {
            for (i, row) in b.iter().enumerate() {
                let w = u[i] / sigma;
                for (vj, bj) in v.iter_mut().zip(row) {
                    *vj += w * bj;
                }
            }
        }
        // svd_flip com decisão por V: a maior coordenada em módulo fica positiva.
        if let Some(&big) = v.iter().max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
            && big < 0.0
        {
            v.iter_mut().for_each(|x| *x = -*x);
        }
        components.push(v);
    }
    components
}

/// Projeta linhas esparsas nos componentes: `X · Vᵀ`.
pub fn project(rows: &[SparseRow], components: &[Vec<f64>]) -> Vec<Vec<f32>> {
    rows.iter().map(|row| components.iter().map(|c| row.iter().map(|&(i, v)| v * c[i]).sum::<f64>() as f32).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_the_dominant_direction() {
        // Linhas em duas direções; a primeira domina (mais linhas e mais peso).
        let x: Vec<SparseRow> =
            vec![vec![(0, 3.0), (1, 3.0)], vec![(0, 2.0), (1, 2.0)], vec![(2, 1.0)], vec![(0, 3.0), (1, 3.1)]];
        let comps = truncated_svd(&x, 3, 2, 10, 5, 0);
        assert_eq!(comps.len(), 2);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((comps[0][0] - h).abs() < 0.01 && (comps[0][1] - h).abs() < 0.01 && comps[0][2].abs() < 1e-6, "{comps:?}");
        assert!((comps[1][2].abs() - 1.0).abs() < 1e-6);
        // Ortonormais.
        assert!(dotv(&comps[0], &comps[1]).abs() < 1e-9);
        assert!((dotv(&comps[0], &comps[0]) - 1.0).abs() < 1e-9);
        // Determinístico.
        assert_eq!(comps, truncated_svd(&x, 3, 2, 10, 5, 0));
    }

    #[test]
    fn symmetric_eigen() {
        let (vals, vecs) = jacobi_eigen(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        assert!((vals[0] - 3.0).abs() < 1e-12 && (vals[1] - 1.0).abs() < 1e-12);
        assert!((vecs[0][0].abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }
}
