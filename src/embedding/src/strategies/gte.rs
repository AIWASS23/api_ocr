//! `gte` — família GTE da Alibaba (`GteEmbedder` do chatbot): `gte-multilingual-base` (70+
//! idiomas, 8192 tokens, 768 dimensões, pooling CLS), `thenlper/gte-*`, `gte-*-en-v1.5`.
//!
//! O `Alibaba-NLP/gte-multilingual-base` usa arquitetura própria (`trust_remote_code` no Python)
//! e não publica ONNX: o grafo vem do espelho `onnx-community/gte-multilingual-base`, e o
//! pooling/normalização do repositório original. Em ONNX não há código remoto para executar.
//!
//! Padrão: `Alibaba-NLP/gte-multilingual-base`. Opções: as do `sentence_transformer`.

use crate::config::EmbeddingConfig;
use crate::error::Result;
use crate::strategies::sentence_transformer::{Prefixes, SentenceTransformerEmbedder};

pub fn load(config: &EmbeddingConfig) -> Result<SentenceTransformerEmbedder> {
    SentenceTransformerEmbedder::load_family(config, "gte", Prefixes::default())
}
