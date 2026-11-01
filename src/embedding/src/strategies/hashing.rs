//! `hashing` — feature hashing do scikit-learn (`HashingVectorizer`, `HashingAdapter` do
//! chatbot): cada termo vai para a coluna `|murmurhash3_32(termo)| mod n_features` e soma a sua
//! contagem. Sem vocabulário, sem ajuste e sem modelo: documentos e consultas podem vir em
//! qualquer ordem, e a mesma configuração dá sempre os mesmos vetores.
//!
//! Padrões do chatbot: `n_features` = `dimensions` ou 4096, `alternate_sign: false`, `norm: null`.
//!
//! Opções: `n_features`, `alternate_sign`, `norm` (`l2`, `l1` ou `null`), `binary` e as do
//! analisador (`analyzer`, `ngram_range`, `lowercase`, `strip_accents`, `token_pattern`,
//! `stop_words`).

use crate::config::EmbeddingConfig;
use crate::core::analyzer::Analyzer;
use crate::core::murmur::hashed_index;
use crate::core::options::Options;
use crate::core::vectorizer::{Norm, SparseRow, densify, normalize_rows, parse_norm};
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

pub struct HashingEmbedder {
    analyzer: Analyzer,
    n_features: usize,
    alternate_sign: bool,
    norm: Option<Norm>,
    binary: bool,
}

pub fn load(config: &EmbeddingConfig) -> Result<HashingEmbedder> {
    let mut o = Options::new("hashing", &config.options);
    let analyzer = Analyzer::from_options(&mut o)?;
    let n_features = o.usize("n_features")?.unwrap_or(config.dimensions.unwrap_or(4096));
    if n_features < 1 {
        return Err(EmbeddingError::config("n_features must be a positive integer"));
    }
    let alternate_sign = o.bool("alternate_sign", false)?;
    let norm = parse_norm(&mut o, None)?;
    let binary = o.bool("binary", false)?;
    o.ignore(&["input", "encoding", "decode_error", "dtype"]);
    o.finish()?;
    Ok(HashingEmbedder { analyzer, n_features, alternate_sign, norm, binary })
}

impl Embedder for HashingEmbedder {
    fn model_name(&self) -> &str {
        "hashing"
    }

    fn embed(&self, texts: &[&str], _input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut rows: Vec<SparseRow> = texts
            .iter()
            .map(|t| {
                let mut acc: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
                for term in self.analyzer.analyze(t) {
                    let (i, sign) = hashed_index(&term, self.n_features);
                    *acc.entry(i).or_insert(0.0) += if self.alternate_sign { sign } else { 1.0 };
                }
                acc.into_iter().map(|(i, v)| (i, if self.binary && v != 0.0 { 1.0 } else { v })).collect()
            })
            .collect();
        if let Some(norm) = self.norm {
            normalize_rows(&mut rows, norm);
        }
        Ok(rows.iter().map(|r| densify(r, self.n_features)).collect())
    }
}
