//! Tokenizadores. As estratégias dependem só do trait [`TextTokenizer`]; a implementação
//! Hugging Face ([`HfTokenizer`], feature `hf-tokenizers`) lê um `tokenizer.json` local ou
//! baixa do Hub (cache em `HF_HOME`), como o `AutoTokenizer.from_pretrained` do chatbot.

use crate::error::Result;

pub trait TextTokenizer: Send + Sync {
    /// Quantidade de tokens SEM tokens especiais (`len(tokenizer.tokenize(text))`).
    fn count(&self, text: &str) -> usize;
    /// Ids de token, com ou sem os especiais (CLS/SEP...).
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;
    /// Texto a partir de ids (sem pular especiais, como o `decode` padrão do transformers).
    fn decode(&self, ids: &[u32]) -> Result<String>;
    /// Limite de tokens do modelo, se conhecido (`max_seq_length` do sentence-transformers).
    fn max_length(&self) -> Option<usize> {
        None
    }
}

#[cfg(feature = "hf-tokenizers")]
pub use hf::HfTokenizer;
#[cfg(feature = "onnx")]
pub(crate) use hf::model_file as hf_model_file;

#[cfg(feature = "hf-tokenizers")]
mod hf {
    use std::path::{Path, PathBuf};

    use super::TextTokenizer;
    use crate::error::{ChunkError, Result};

    pub struct HfTokenizer {
        inner: tokenizers::Tokenizer,
        max_length: Option<usize>,
        name: String,
    }

    /// Caminho local de um arquivo do modelo: diretório local, arquivo local ou download do Hub.
    pub(crate) fn model_file(name_or_path: &str, file: &str) -> Result<PathBuf> {
        let p = Path::new(name_or_path);
        if p.is_file() {
            return Ok(if p.file_name().is_some_and(|f| f == file) { p.to_path_buf() } else { p.with_file_name(file) });
        }
        if p.is_dir() {
            return Ok(p.join(file));
        }
        let api = hf_hub::api::sync::Api::new().map_err(|e| ChunkError::Model(format!("Hugging Face Hub: {e}")))?;
        api.model(name_or_path.to_string())
            .get(file)
            .map_err(|e| ChunkError::Model(format!("'{name_or_path}/{file}': {e}")))
    }

    impl HfTokenizer {
        /// Carrega `tokenizer.json` (local ou do Hub) e, se houver, o `max_seq_length` de
        /// `sentence_bert_config.json`.
        pub fn load(name_or_path: &str) -> Result<Self> {
            let path = model_file(name_or_path, "tokenizer.json")?;
            let mut inner = tokenizers::Tokenizer::from_file(&path)
                .map_err(|e| ChunkError::Model(format!("Erro ao carregar modelo '{name_or_path}': {e}")))?;
            // Sem truncamento nem padding: contagens precisam ser do texto inteiro.
            inner
                .with_truncation(None)
                .map_err(|e| ChunkError::Model(e.to_string()))?
                .with_padding(None);
            let max_length = model_file(name_or_path, "sentence_bert_config.json")
                .ok()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v.get("max_seq_length").and_then(|m| m.as_u64()))
                .map(|m| m as usize);
            Ok(Self { inner, max_length, name: name_or_path.to_string() })
        }

        pub fn from_tokenizer(inner: tokenizers::Tokenizer, max_length: Option<usize>) -> Self {
            Self { inner, max_length, name: String::new() }
        }

        pub fn name(&self) -> &str {
            &self.name
        }

        pub fn inner(&self) -> &tokenizers::Tokenizer {
            &self.inner
        }
    }

    impl TextTokenizer for HfTokenizer {
        fn count(&self, text: &str) -> usize {
            self.inner.encode_fast(text, false).map(|e| e.len()).unwrap_or(0)
        }

        fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
            self.inner
                .encode_fast(text, add_special_tokens)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| ChunkError::Model(e.to_string()))
        }

        fn decode(&self, ids: &[u32]) -> Result<String> {
            self.inner.decode(ids, false).map_err(|e| ChunkError::Model(e.to_string()))
        }

        fn max_length(&self) -> Option<usize> {
            self.max_length
        }
    }
}
