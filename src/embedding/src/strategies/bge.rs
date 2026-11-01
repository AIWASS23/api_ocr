//! `bge` — família BGE da BAAI (`BgeEmbedder` do chatbot): `bge-m3` (multilíngue, 8192 tokens,
//! 1024 dimensões), `bge-small/base/large-en-v1.5`, `bge-*-zh-v1.5`. Pooling pelo token CLS e
//! vetores normalizados, como manda o `modules.json` de cada modelo.
//!
//! Sem prefixo por padrão, como no Python. Os modelos `*-v1.5` recomendam, só nas consultas
//! curtas, `options.query_prefix = "Represent this sentence for searching relevant passages: "`.
//!
//! Padrão: `BAAI/bge-m3` (o ONNX tem ~2,2 GB em dados externos). Opções: as do
//! `sentence_transformer`.

use crate::config::EmbeddingConfig;
use crate::error::Result;
use crate::strategies::sentence_transformer::{Prefixes, SentenceTransformerEmbedder};

pub fn load(config: &EmbeddingConfig) -> Result<SentenceTransformerEmbedder> {
    SentenceTransformerEmbedder::load_family(config, "bge", Prefixes::default())
}
