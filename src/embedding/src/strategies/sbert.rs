//! `sbert` — modelos SBERT clássicos (`SBertEmbedder` do chatbot): MiniLM, MPNet, KR-SBERT...
//! Mesmo pipeline do `sentence_transformer`, sem prefixos.
//!
//! Padrão: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensões). Opções: as do
//! `sentence_transformer`.

use crate::config::EmbeddingConfig;
use crate::error::Result;
use crate::strategies::sentence_transformer::{Prefixes, SentenceTransformerEmbedder};

pub fn load(config: &EmbeddingConfig) -> Result<SentenceTransformerEmbedder> {
    SentenceTransformerEmbedder::load_family(config, "sbert", Prefixes::default())
}
