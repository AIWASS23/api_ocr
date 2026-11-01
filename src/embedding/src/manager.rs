//! `EmbeddingManager` — uma configuração e um método para qualquer estratégia (o equivalente de
//! `embedding_manager.py`).
//!
//! ```no_run
//! use embedding::{EmbeddingManager, Embedder};
//! use serde_json::json;
//!
//! let m = EmbeddingManager::from_value(json!({"strategy": "bge", "model_name": "BAAI/bge-m3", "normalize": true}))?;
//! let docs = m.embed_documents(&["documento A", "documento B"])?;
//! let query = m.embed_query("minha consulta")?;
//! # Ok::<(), embedding::EmbeddingError>(())
//! ```
//!
//! O modelo é carregado no primeiro uso (ou em [`EmbeddingManager::load`]). Toda saída passa
//! pelo `_finalize` do Python: uma linha por texto, valores finitos, corte em `dimensions` e
//! normalização opcional.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::Value;

use crate::config::{EmbeddingConfig, SUPPORTED_STRATEGIES};
use crate::core::matrix::finalize;
use crate::embedder::{Embedder, ImageInput, InputType, SparseEmbedding};
use crate::error::{EmbeddingError, Result};
#[allow(unused_imports)]
use crate::strategies as s;

pub struct EmbeddingManager {
    config: EmbeddingConfig,
    embedder: Mutex<Option<Arc<dyn Embedder>>>,
    /// Largura do último resultado (0 = ainda não se sabe).
    dimension: AtomicUsize,
}

#[allow(unused_macros)]
macro_rules! boxed {
    ($e:expr) => {
        Ok(Arc::new($e?) as Arc<dyn Embedder>)
    };
}

/// Constrói a estratégia da configuração.
fn build(config: &EmbeddingConfig) -> Result<Arc<dyn Embedder>> {
    let strategy = config.strategy.as_str();
    // `embed_options` só vale para o Ollama; nas locais o Python também o ignora.
    let mut config = config.clone();
    if strategy != "ollama" {
        config.options.remove("embed_options");
    }
    let config = &config;
    match strategy {
        "tfidf" => boxed!(s::tfidf::load(config)),
        "hashing" => boxed!(s::hashing::load(config)),
        "bm25" => boxed!(s::bm25::load(config)),
        "lsa" => boxed!(s::lsa::load(config)),
        "custom" => Err(EmbeddingError::config("custom strategy requires an injected embedder (EmbeddingManager::with_embedder)")),
        #[cfg(feature = "onnx")]
        "sentence_transformer" => boxed!(s::sentence_transformer::load(config)),
        #[cfg(feature = "onnx")]
        "sbert" => boxed!(s::sbert::load(config)),
        #[cfg(feature = "onnx")]
        "bge" => boxed!(s::bge::load(config)),
        #[cfg(feature = "onnx")]
        "gte" => boxed!(s::gte::load(config)),
        #[cfg(feature = "onnx")]
        "e5" => boxed!(s::e5::load(config)),
        #[cfg(feature = "onnx")]
        "multilingual" => boxed!(s::multilingual::load(config)),
        #[cfg(feature = "onnx")]
        "instructor" => boxed!(s::instructor::load(config)),
        #[cfg(feature = "onnx")]
        "jina" => boxed!(s::jina::load(config)),
        #[cfg(feature = "onnx")]
        "nomic" => boxed!(s::nomic::load(config)),
        #[cfg(feature = "onnx")]
        "huggingface" => boxed!(s::huggingface::load(config)),
        #[cfg(feature = "onnx")]
        "sparse" => boxed!(s::sparse::load(config)),
        #[cfg(feature = "vision")]
        "clip" => boxed!(s::clip::load(config)),
        #[cfg(feature = "vision")]
        "siglip" => boxed!(s::siglip::load(config)),
        #[cfg(feature = "ollama")]
        "ollama" => boxed!(s::ollama::load(config)),
        #[cfg(not(feature = "vision"))]
        "clip" => Err(EmbeddingError::FeatureDisabled { strategy: "clip", feature: "vision" }),
        #[cfg(not(feature = "vision"))]
        "siglip" => Err(EmbeddingError::FeatureDisabled { strategy: "siglip", feature: "vision" }),
        #[cfg(not(feature = "ollama"))]
        "ollama" => Err(EmbeddingError::FeatureDisabled { strategy: "ollama", feature: "ollama" }),
        other => {
            let name = SUPPORTED_STRATEGIES.iter().find(|s| **s == other).copied().unwrap_or("?");
            Err(EmbeddingError::FeatureDisabled { strategy: name, feature: "onnx" })
        }
    }
}

impl EmbeddingManager {
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        Ok(Self { config: config.validated()?, embedder: Mutex::new(None), dimension: AtomicUsize::new(0) })
    }

    /// A partir do `embedding:` do YAML do chatbot.
    pub fn from_value(value: Value) -> Result<Self> {
        Self::new(EmbeddingConfig::from_value(value)?)
    }

    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Nomes canônicos aceitos.
    pub fn strategies() -> &'static [&'static str] {
        SUPPORTED_STRATEGIES
    }

    /// Embedder externo no lugar do da estratégia (obrigatório na `custom`; o `options.embedder`
    /// do Python).
    pub fn with_embedder(self, embedder: Arc<dyn Embedder>) -> Self {
        *self.embedder.lock().unwrap_or_else(PoisonError::into_inner) = Some(embedder);
        self
    }

    pub fn is_loaded(&self) -> bool {
        self.embedder.lock().unwrap_or_else(PoisonError::into_inner).is_some()
    }

    /// Carrega (uma vez) e devolve o embedder da estratégia.
    pub fn load(&self) -> Result<Arc<dyn Embedder>> {
        let mut slot = self.embedder.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = slot.as_ref() {
            return Ok(e.clone());
        }
        let built = build(&self.config)?;
        *slot = Some(built.clone());
        Ok(built)
    }

    /// Dimensão dos vetores, depois do primeiro uso.
    pub fn dimension(&self) -> Option<usize> {
        Some(self.dimension.load(Ordering::Relaxed)).filter(|&d| d > 0)
    }

    fn finish(&self, values: Vec<Vec<f32>>, rows: usize) -> Result<Vec<Vec<f32>>> {
        let out = finalize(values, rows, self.config.dimensions, self.config.normalize)?;
        if let Some(first) = out.first() {
            self.dimension.store(first.len(), Ordering::Relaxed);
        }
        Ok(out)
    }

    /// Embeddings de documentos ou consultas, já finalizados.
    pub fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let values = self.load()?.embed(texts, input_type)?;
        self.finish(values, texts.len())
    }
}

impl Embedder for EmbeddingManager {
    fn model_name(&self) -> &str {
        self.config.model_or_strategy()
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        EmbeddingManager::embed(self, texts, input_type)
    }

    fn supports_images(&self) -> bool {
        self.load().is_ok_and(|e| e.supports_images())
    }

    fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let embedder = self.load()?;
        if !embedder.supports_images() {
            return Err(EmbeddingError::ImagesUnsupported(self.config.strategy.clone()));
        }
        let values = embedder.embed_images(images)?;
        self.finish(values, images.len())
    }

    fn embed_sparse(&self, texts: &[&str], input_type: InputType) -> Result<Vec<SparseEmbedding>> {
        Ok(EmbeddingManager::embed(self, texts, input_type)?.iter().map(|v| SparseEmbedding::from_dense(v)).collect())
    }
}
