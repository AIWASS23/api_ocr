//! `tfidf` — TF-IDF do scikit-learn (`TfidfVectorizer`, `TfidfAdapter` do chatbot): o primeiro
//! lote de documentos define o vocabulário (os `max_features` termos mais frequentes, em ordem
//! alfabética) e o IDF; depois, documentos e consultas são projetados nele. Sem modelo.
//!
//! Consultar antes de embutir documentos é erro, como no Python. Vetor do tamanho do
//! vocabulário (`dimensions` ou 4096 termos no máximo).
//!
//! Opções (as do `TfidfVectorizer`): `analyzer` (`word`, `char`, `char_wb`), `ngram_range`,
//! `lowercase`, `strip_accents` (`ascii`, `unicode`), `token_pattern`, `stop_words` (`english` ou
//! lista), `max_df`, `min_df`, `max_features`, `vocabulary`, `binary`, `norm` (`l2`, `l1` ou
//! `null`), `use_idf`, `smooth_idf`, `sublinear_tf`.
//!
//! Diferença conhecida: com empate de frequência exatamente no corte de `max_features`, os
//! termos mantidos podem diferir dos do NumPy (que não ordena de forma estável); aqui o empate
//! fica com a ordem alfabética.

use std::sync::{Mutex, PoisonError};

use crate::config::EmbeddingConfig;
use crate::core::options::Options;
use crate::core::vectorizer::{CountVectorizer, TfidfWeights, Vocabulary, densify};
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

pub struct TfidfEmbedder {
    vectorizer: CountVectorizer,
    template: TfidfWeights,
    fitted: Mutex<Option<(Vocabulary, TfidfWeights)>>,
}

pub fn load(config: &EmbeddingConfig) -> Result<TfidfEmbedder> {
    let mut o = Options::new("tfidf", &config.options);
    let vectorizer = CountVectorizer::from_options(&mut o, Some(config.dimensions.unwrap_or(4096)))?;
    let template = TfidfWeights::from_options(&mut o)?;
    o.ignore(&["input", "encoding", "decode_error", "dtype"]);
    o.finish()?;
    Ok(TfidfEmbedder { vectorizer, template, fitted: Mutex::new(None) })
}

impl TfidfEmbedder {
    /// Termos do vocabulário aprendido (coluna → termo).
    pub fn vocabulary(&self) -> Option<Vec<String>> {
        self.fitted.lock().unwrap_or_else(PoisonError::into_inner).as_ref().map(|(v, _)| v.terms.clone())
    }
}

impl Embedder for TfidfEmbedder {
    fn model_name(&self) -> &str {
        "tfidf"
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut fitted = self.fitted.lock().unwrap_or_else(PoisonError::into_inner);
        let rows = match fitted.as_ref() {
            None => {
                if input_type == InputType::Query {
                    return Err(EmbeddingError::input("TF-IDF must embed documents before embedding queries"));
                }
                let (vocab, counts) = self.vectorizer.fit_transform(texts)?;
                let mut weights = self.template.clone();
                weights.fit(&counts, vocab.len());
                let rows = weights.transform(counts);
                *fitted = Some((vocab, weights));
                rows
            }
            Some((vocab, weights)) => weights.transform(self.vectorizer.transform(vocab, texts)),
        };
        let width = fitted.as_ref().map_or(0, |(v, _)| v.len());
        Ok(rows.iter().map(|r| densify(r, width)).collect())
    }
}
