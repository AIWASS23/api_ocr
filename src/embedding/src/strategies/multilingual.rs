//! `multilingual` — modelos multilíngues do sentence-transformers (`MultilingualEmbeddings` do
//! chatbot): `paraphrase-multilingual-MiniLM-L12-v2`, `distiluse-base-multilingual-cased-v1/v2`,
//! `LaBSE` (estes dois com camada `Dense`, lida do `safetensors` ou do `pytorch_model.bin`).
//!
//! Padrão: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões, 50+
//! idiomas, inclusive português). Opções: as do `sentence_transformer`.

use crate::config::EmbeddingConfig;
use crate::error::Result;
use crate::strategies::sentence_transformer::{Prefixes, SentenceTransformerEmbedder};

pub fn load(config: &EmbeddingConfig) -> Result<SentenceTransformerEmbedder> {
    SentenceTransformerEmbedder::load_family(config, "multilingual", Prefixes::default())
}
