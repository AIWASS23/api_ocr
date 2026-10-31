//! Embeddings de texto para a estratégia `semantic`. Qualquer [`Embedder`] serve; o
//! [`FastEmbedder`] (feature `embeddings`) roda o modelo ONNX localmente, em processo, via
//! `fastembed` + ONNX Runtime (ex.: `BAAI/bge-m3`, `intfloat/multilingual-e5-large`).

use crate::error::Result;

pub trait Embedder: Send + Sync {
    /// Um vetor por texto, na mesma ordem (`embeddings.embed_documents`).
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[cfg(feature = "embeddings")]
pub use local::FastEmbedder;

#[cfg(feature = "embeddings")]
mod local {
    use std::sync::Mutex;

    use fastembed::{TextEmbedding, TextInitOptions};

    use super::Embedder;
    use crate::error::{ChunkError, Result};

    /// Cache dos modelos: `FASTEMBED_CACHE_DIR`, senão `$XDG_CACHE_HOME/fastembed` ou
    /// `~/.cache/fastembed` (o padrão do fastembed seria o diretório atual).
    fn cache_dir() -> std::path::PathBuf {
        if let Some(d) = std::env::var_os("FASTEMBED_CACHE_DIR") {
            return d.into();
        }
        let base = std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))
            .unwrap_or_else(std::env::temp_dir);
        base.join("fastembed")
    }

    pub struct FastEmbedder {
        model: Mutex<TextEmbedding>,
    }

    impl FastEmbedder {
        /// `model_name` no formato do Hugging Face (`BAAI/bge-m3`, `intfloat/multilingual-e5-large`...).
        /// O fastembed cataloga muitos modelos pelo espelho ONNX (`Xenova/...`, `Qdrant/...-onnx`):
        /// além do nome exato, aceita o mesmo modelo publicado por outro dono, preferindo a
        /// versão não quantizada.
        pub fn load(model_name: &str) -> Result<Self> {
            let supported = TextEmbedding::list_supported_models();
            let base = |code: &str| {
                let last = code.rsplit('/').next().unwrap_or(code).to_ascii_lowercase();
                
                last.strip_suffix("-onnx-q").or_else(|| last.strip_suffix("-onnx")).unwrap_or(&last).to_string()
            };
            let wanted = base(model_name);
            let quantized = |code: &str| code.to_ascii_lowercase().ends_with("-q");
            let info = supported
                .iter()
                .find(|m| m.model_code.eq_ignore_ascii_case(model_name))
                .or_else(|| supported.iter().find(|m| base(&m.model_code) == wanted && !quantized(&m.model_code)))
                .or_else(|| supported.iter().find(|m| base(&m.model_code) == wanted))
                .ok_or_else(|| {
                    let names: Vec<&str> = supported.iter().map(|m| m.model_code.as_str()).collect();
                    ChunkError::Model(format!("modelo de embeddings '{model_name}' não suportado. Disponíveis: {}", names.join(", ")))
                })?;
            let environment = docling_onnx::environment().map_err(ChunkError::Model)?;
            let options = TextInitOptions::new(info.model.clone()).with_cache_dir(cache_dir()).with_environment(environment.clone());
            let model = TextEmbedding::try_new(options)
                .map_err(|e| ChunkError::Model(format!("Erro ao carregar modelo '{model_name}': {e}")))?;
            Ok(Self { model: Mutex::new(model) })
        }
    }

    impl Embedder for FastEmbedder {
        fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut model = self.model.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            model.embed(texts, None).map_err(|e| ChunkError::Model(e.to_string()))
        }
    }
}
