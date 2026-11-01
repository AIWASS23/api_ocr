//! `lsa` (nova, só lógica) — Análise Semântica Latente: TF-IDF seguido de SVD truncado (o
//! `TruncatedSVD` do scikit-learn, algoritmo aleatorizado). O primeiro lote de documentos define
//! o vocabulário, o IDF e os `n` eixos de maior variância; cada texto vira um vetor DENSO de `n`
//! dimensões, em que termos que aparecem nos mesmos contextos ficam próximos (sinônimos do
//! próprio acervo), sem modelo nem rede.
//!
//! Precisa de um acervo razoável no ajuste (centenas de documentos ou mais); as dimensões ficam
//! limitadas ao número de documentos e de termos. Determinístico (`random_state`), ao contrário
//! do padrão do scikit-learn.
//!
//! Opções: `n_components` (padrão: `dimensions` ou 100), `n_iter` (5), `n_oversamples` (10),
//! `random_state` (0) e as do `tfidf` (analisador, vocabulário e pesos).

use std::sync::{Mutex, PoisonError};

use crate::config::EmbeddingConfig;
use crate::core::options::Options;
use crate::core::svd::{project, truncated_svd};
use crate::core::vectorizer::{CountVectorizer, TfidfWeights, Vocabulary};
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

struct Fitted {
    vocab: Vocabulary,
    weights: TfidfWeights,
    components: Vec<Vec<f64>>,
}

pub struct LsaEmbedder {
    vectorizer: CountVectorizer,
    template: TfidfWeights,
    n_components: usize,
    n_iter: usize,
    n_oversamples: usize,
    seed: u64,
    fitted: Mutex<Option<Fitted>>,
}

pub fn load(config: &EmbeddingConfig) -> Result<LsaEmbedder> {
    let mut o = Options::new("lsa", &config.options);
    let n_components = o.usize("n_components")?.or(config.dimensions).unwrap_or(100);
    if n_components < 1 {
        return Err(EmbeddingError::config("lsa: n_components must be a positive integer"));
    }
    let n_iter = o.usize("n_iter")?.unwrap_or(5);
    let n_oversamples = o.usize("n_oversamples")?.unwrap_or(10);
    let seed = o.usize("random_state")?.unwrap_or(0) as u64;
    let vectorizer = CountVectorizer::from_options(&mut o, None)?;
    let template = TfidfWeights::from_options(&mut o)?;
    o.ignore(&["input", "encoding", "decode_error", "dtype", "algorithm", "power_iteration_normalizer"]);
    o.finish()?;
    Ok(LsaEmbedder { vectorizer, template, n_components, n_iter, n_oversamples, seed, fitted: Mutex::new(None) })
}

impl Embedder for LsaEmbedder {
    fn model_name(&self) -> &str {
        "lsa"
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut fitted = self.fitted.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(f) = fitted.as_ref() {
            let rows = f.weights.transform(self.vectorizer.transform(&f.vocab, texts));
            return Ok(project(&rows, &f.components));
        }
        if input_type == InputType::Query {
            return Err(EmbeddingError::input("LSA must embed documents before embedding queries"));
        }
        let (vocab, counts) = self.vectorizer.fit_transform(texts)?;
        let mut weights = self.template.clone();
        weights.fit(&counts, vocab.len());
        let rows = weights.transform(counts);
        let components = truncated_svd(&rows, vocab.len(), self.n_components, self.n_oversamples, self.n_iter, self.seed);
        let out = project(&rows, &components);
        *fitted = Some(Fitted { vocab, weights, components });
        Ok(out)
    }
}
