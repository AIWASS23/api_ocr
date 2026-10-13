//! `token_sentence_transformer` — janelas de tokens dimensionadas ao contexto de um modelo
//! Sentence Transformers (`SentenceTransformersTokenTextSplitter`): tokens por janela =
//! `min(chunk_size, max_seq_length do modelo)`. Carrega o tokenizador sozinho (como no chatbot).
//!
//! Opções: `model_name` (padrão: o `model_name` da configuração).

use std::sync::Arc;

use crate::chunker::{Chunker, Sizing};
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::tokens::split_text_on_tokens;
use crate::error::{ChunkError, Result};
use crate::models::tokenizer::TextTokenizer;

pub struct ChunkSentenceTransformerToken {
    sizing: Sizing,
    tokenizer: Arc<dyn TextTokenizer>,
    tokens_per_chunk: usize,
}

impl ChunkSentenceTransformerToken {
    pub fn new(tokenizer: Arc<dyn TextTokenizer>, model_name: &str, sizing: Sizing) -> Result<Self> {
        let max = tokenizer.max_length().ok_or_else(|| {
            ChunkError::config(
                "The model does not have a maximum token limit, and tokens_per_chunk was not provided. \
                 Please provide a value for tokens_per_chunk.",
            )
        })?;
        let tokens_per_chunk = sizing.chunk_size.min(max);
        if sizing.chunk_overlap >= tokens_per_chunk {
            return Err(ChunkError::config(format!(
                "chunk_overlap must be smaller than the model's effective token limit ({tokens_per_chunk})"
            )));
        }
        let _ = model_name;
        Ok(Self { sizing, tokenizer, tokens_per_chunk })
    }

    pub fn tokens_per_chunk(&self) -> usize {
        self.tokens_per_chunk
    }
}

impl Chunker for ChunkSentenceTransformerToken {
    crate::sizing_accessors!("token_sentence_transformer");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        // Ids com os tokens de início/fim, que são descartados (`[1:-1]`).
        let ids = self.tokenizer.encode(text, true)?;
        let inner = if ids.len() >= 2 { &ids[1..ids.len() - 1] } else { &[][..] };
        let chunks =
            split_text_on_tokens(inner, self.tokens_per_chunk, self.sizing.chunk_overlap, |w| self.tokenizer.decode(w))?;
        Ok(normalize_chunks(chunks))
    }
}
