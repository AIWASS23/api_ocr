//! `e5` — família E5 da intfloat (`multilingual-e5-small/base/large`, `e5-*-v2`): o texto leva o
//! prefixo `query: ` nas consultas e `passage: ` nos documentos, como o modelo foi treinado (o
//! `_prepare` do adaptador Python).
//!
//! Padrão: `intfloat/multilingual-e5-base` (768 dimensões). Opções: as do
//! `sentence_transformer`; `query_prefix`/`document_prefix` trocam os prefixos.

use crate::config::EmbeddingConfig;
use crate::error::Result;
use crate::strategies::sentence_transformer::{Prefixes, SentenceTransformerEmbedder};

pub const PREFIXES: Prefixes = Prefixes { query: "query: ", document: "passage: " };

pub fn load(config: &EmbeddingConfig) -> Result<SentenceTransformerEmbedder> {
    SentenceTransformerEmbedder::load_family(config, "e5", PREFIXES)
}
