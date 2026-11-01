//! `nomic` — Nomic Embed (`nomic-embed-text-v1/v1.5`): prefixos `search_query: ` nas consultas e
//! `search_document: ` nos documentos (o `_prepare` do adaptador Python). O v1.5 é Matryoshka:
//! `dimensions` (64 a 768) corta o vetor; combine com `normalize: true`.
//!
//! Padrão: `nomic-ai/nomic-embed-text-v1.5` (768 dimensões, 8192 tokens). Opções: as do
//! `sentence_transformer`.

use crate::config::EmbeddingConfig;
use crate::error::Result;
use crate::strategies::sentence_transformer::{Prefixes, SentenceTransformerEmbedder};

pub const PREFIXES: Prefixes = Prefixes { query: "search_query: ", document: "search_document: " };

pub fn load(config: &EmbeddingConfig) -> Result<SentenceTransformerEmbedder> {
    SentenceTransformerEmbedder::load_family(config, "nomic", PREFIXES)
}
