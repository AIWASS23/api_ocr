//! O módulo `Pooling` do sentence-transformers: de vetores por token para um vetor por texto.
//! Modos `cls`, `max`, `mean`, `mean_sqrt_len_tokens`, `weightedmean` e `lasttoken`
//! (concatenados, nessa ordem, quando há vários) e `include_prompt=false`.

use serde_json::Value;

use crate::error::{EmbeddingError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingMode {
    Cls,
    Max,
    Mean,
    MeanSqrtLen,
    WeightedMean,
    LastToken,
}

impl PoolingMode {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "cls" => PoolingMode::Cls,
            "max" => PoolingMode::Max,
            "mean" => PoolingMode::Mean,
            "mean_sqrt_len_tokens" => PoolingMode::MeanSqrtLen,
            "weightedmean" => PoolingMode::WeightedMean,
            "lasttoken" => PoolingMode::LastToken,
            other => return Err(EmbeddingError::config(format!("Invalid pooling mode: '{other}'"))),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pooling {
    pub modes: Vec<PoolingMode>,
    /// `false`: os tokens do prompt ficam fora do pooling (INSTRUCTOR).
    pub include_prompt: bool,
}

impl Default for Pooling {
    fn default() -> Self {
        Self { modes: vec![PoolingMode::Mean], include_prompt: true }
    }
}

/// Chaves antigas, na ordem de concatenação do sentence-transformers.
const LEGACY: &[(&str, PoolingMode)] = &[
    ("pooling_mode_cls_token", PoolingMode::Cls),
    ("pooling_mode_max_tokens", PoolingMode::Max),
    ("pooling_mode_mean_tokens", PoolingMode::Mean),
    ("pooling_mode_mean_sqrt_len_tokens", PoolingMode::MeanSqrtLen),
    ("pooling_mode_weightedmean_tokens", PoolingMode::WeightedMean),
    ("pooling_mode_lasttoken", PoolingMode::LastToken),
];

impl Pooling {
    /// `1_Pooling/config.json`, no formato novo (`pooling_mode`) ou no antigo (`pooling_mode_*`).
    pub fn from_config(config: &Value) -> Result<Self> {
        let include_prompt = config.get("include_prompt").and_then(Value::as_bool).unwrap_or(true);
        let modes = match config.get("pooling_mode") {
            Some(Value::String(s)) => vec![PoolingMode::parse(s)?],
            Some(Value::Array(a)) => a.iter().map(|m| PoolingMode::parse(m.as_str().unwrap_or(""))).collect::<Result<_>>()?,
            _ => {
                let active: Vec<PoolingMode> =
                    LEGACY.iter().filter(|(k, _)| config.get(*k).and_then(Value::as_bool).unwrap_or(false)).map(|(_, m)| *m).collect();
                if active.is_empty() { vec![PoolingMode::Mean] } else { active }
            }
        };
        Ok(Self { modes, include_prompt })
    }

    /// Um vetor a partir de `tokens` (`seq × hidden`, por linha) e da máscara de atenção.
    /// `prompt_length`: tokens iniciais do prompt (só contam com `include_prompt=false`).
    pub fn pool(&self, tokens: &[f32], seq: usize, hidden: usize, mask: &[i64], prompt_length: Option<usize>) -> Vec<f32> {
        let mut mask: Vec<f32> = mask.iter().map(|&m| if m != 0 { 1.0 } else { 0.0 }).collect();
        if !self.include_prompt
            && let Some(pl) = prompt_length
        {
            // Sem padding à esquerda: zera os primeiros `pl`; com, zera o padding e o prompt.
            let first = mask.iter().position(|&m| m != 0.0).unwrap_or(0);
            mask.iter_mut().take(first + pl).for_each(|m| *m = 0.0);
        }
        let row = |t: usize| &tokens[t * hidden..(t + 1) * hidden];
        let mut out = Vec::with_capacity(hidden * self.modes.len());
        for mode in &self.modes {
            match mode {
                PoolingMode::Cls => {
                    let first = mask.iter().position(|&m| m != 0.0).unwrap_or(0);
                    out.extend_from_slice(row(first));
                }
                PoolingMode::Max => {
                    let mut v = vec![f32::NEG_INFINITY; hidden];
                    for t in (0..seq).filter(|&t| mask[t] != 0.0) {
                        for (a, &b) in v.iter_mut().zip(row(t)) {
                            *a = a.max(b);
                        }
                    }
                    out.extend(v);
                }
                PoolingMode::Mean | PoolingMode::MeanSqrtLen => {
                    let mut sum = vec![0f32; hidden];
                    for (t, &m) in mask.iter().enumerate().take(seq) {
                        if m != 0.0 {
                            for (a, &b) in sum.iter_mut().zip(row(t)) {
                                *a += b * m;
                            }
                        }
                    }
                    let count = mask.iter().sum::<f32>().max(1e-9);
                    let div = if *mode == PoolingMode::Mean { count } else { count.sqrt() };
                    out.extend(sum.into_iter().map(|x| x / div));
                }
                PoolingMode::WeightedMean => {
                    let mut sum = vec![0f32; hidden];
                    let mut weights = 0f32;
                    for (t, &m) in mask.iter().enumerate().take(seq) {
                        let w = m * (t + 1) as f32;
                        if w != 0.0 {
                            weights += w;
                            for (a, &b) in sum.iter_mut().zip(row(t)) {
                                *a += b * w;
                            }
                        }
                    }
                    let w = weights.max(1e-9);
                    out.extend(sum.into_iter().map(|x| x / w));
                }
                PoolingMode::LastToken => {
                    let last = mask.iter().rposition(|&m| m != 0.0);
                    match last {
                        Some(t) => out.extend_from_slice(row(t)),
                        None => out.extend(std::iter::repeat_n(0.0, hidden)),
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // 3 tokens de dimensão 2; o último é padding.
    const T: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 100.0, 100.0];
    const M: [i64; 3] = [1, 1, 0];

    fn pool(modes: &[PoolingMode]) -> Vec<f32> {
        Pooling { modes: modes.to_vec(), include_prompt: true }.pool(&T, 3, 2, &M, None)
    }

    #[test]
    fn modes_ignore_padding() {
        assert_eq!(pool(&[PoolingMode::Mean]), vec![2.0, 3.0]);
        assert_eq!(pool(&[PoolingMode::Cls]), vec![1.0, 2.0]);
        assert_eq!(pool(&[PoolingMode::Max]), vec![3.0, 4.0]);
        assert_eq!(pool(&[PoolingMode::LastToken]), vec![3.0, 4.0]);
        let s = 2f32.sqrt();
        assert_eq!(pool(&[PoolingMode::MeanSqrtLen]), vec![4.0 / s, 6.0 / s]);
        // pesos 1 e 2: (1*1 + 2*3)/3, (1*2 + 2*4)/3
        assert_eq!(pool(&[PoolingMode::WeightedMean]), vec![7.0 / 3.0, 10.0 / 3.0]);
        assert_eq!(pool(&[PoolingMode::Cls, PoolingMode::Mean]), vec![1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn prompt_tokens_can_be_excluded() {
        let p = Pooling { modes: vec![PoolingMode::Mean], include_prompt: false };
        assert_eq!(p.pool(&T, 3, 2, &M, Some(1)), vec![3.0, 4.0]);
        assert_eq!(Pooling::default().pool(&T, 3, 2, &M, Some(1)), vec![2.0, 3.0]);
    }

    #[test]
    fn config_formats() {
        let legacy = json!({"word_embedding_dimension": 4, "pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true});
        assert_eq!(Pooling::from_config(&legacy).unwrap().modes, vec![PoolingMode::Cls, PoolingMode::Mean]);
        let new = json!({"embedding_dimension": 4, "pooling_mode": "lasttoken", "include_prompt": false});
        let p = Pooling::from_config(&new).unwrap();
        assert_eq!((p.modes, p.include_prompt), (vec![PoolingMode::LastToken], false));
        assert_eq!(Pooling::from_config(&json!({})).unwrap().modes, vec![PoolingMode::Mean]);
        assert!(Pooling::from_config(&json!({"pooling_mode": "median"})).is_err());
    }
}
