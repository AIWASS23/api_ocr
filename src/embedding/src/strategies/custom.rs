//! `custom` — embedder fornecido pelo usuário (o `options.embedder` do Python): qualquer tipo que
//! implemente [`Embedder`], injetado com [`crate::EmbeddingManager::with_embedder`], ou uma
//! função `(textos, tipo) -> vetores` com [`from_fn`]. O manager completa o contrato (consulta,
//! documentos, esparso, assíncrono) e aplica `dimensions`/`normalize`.

use std::sync::Arc;

use crate::embedder::{Embedder, InputType};
use crate::error::Result;

type EmbedFn = dyn Fn(&[&str], InputType) -> Result<Vec<Vec<f32>>> + Send + Sync;

pub struct FnEmbedder {
    name: String,
    f: Box<EmbedFn>,
}

impl Embedder for FnEmbedder {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        (self.f)(texts, input_type)
    }
}

/// Embedder a partir de uma função.
pub fn from_fn<F>(name: impl Into<String>, f: F) -> Arc<dyn Embedder>
where
    F: Fn(&[&str], InputType) -> Result<Vec<Vec<f32>>> + Send + Sync + 'static,
{
    Arc::new(FnEmbedder { name: name.into(), f: Box::new(f) })
}
