//! `contextual_retrieval` (nova) — *Contextual Retrieval* (Anthropic): cada chunk de uma
//! estratégia base recebe, antes do texto, uma ou duas frases escritas por um LLM que situam o
//! trecho no documento ("Este trecho é da seção de taxas do código tributário de 2023..."). O
//! embedding passa a carregar o contexto que o trecho isolado não tem.
//!
//! Custo: uma chamada de LLM por chunk na indexação. O documento enviado ao LLM é truncado em
//! `max_document_chars`. O texto final passa do `chunk_size` pelo tamanho do contexto.
//!
//! Opções: `base_strategy` (`"recursive"`), `max_document_chars` (24000), `separator`
//! (`"\n\n"`), `fail_open` (`false`: com `true`, um erro do LLM deixa o chunk sem contexto em
//! vez de falhar), mais as do cliente LLM (ver `ChunkManager`). Metadados: `context`,
//! `original_chunk`.

use std::sync::Arc;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::strip;
use crate::document::Document;
use crate::error::Result;
use crate::models::llm::ContextModel;

pub struct ChunkContextualRetrieval {
    sizing: Sizing,
    base: Box<dyn Chunker>,
    model: Arc<dyn ContextModel>,
    max_document_chars: usize,
    separator: String,
    fail_open: bool,
}

impl ChunkContextualRetrieval {
    /// `o` já sem `base_strategy` e sem as opções do cliente LLM (o manager as retira).
    pub fn new(sizing: Sizing, base: Box<dyn Chunker>, model: Arc<dyn ContextModel>, mut o: Options) -> Result<Self> {
        let max_document_chars = o.usize("max_document_chars")?.unwrap_or(24_000);
        let separator = o.string("separator")?.unwrap_or_else(|| "\n\n".into());
        let fail_open = o.bool("fail_open", false)?;
        o.finish()?;
        Ok(Self { sizing, base, model, max_document_chars, separator, fail_open })
    }
}

impl Chunker for ChunkContextualRetrieval {
    crate::sizing_accessors!("contextual_retrieval");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.chunk_documents(text)?.into_iter().map(|d| d.page_content).collect())
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let document: String = text.chars().take(self.max_document_chars).collect();
        let mut out = Vec::new();
        for chunk in self.base.chunk_documents(text)? {
            let context = match self.model.situate(&document, &chunk.page_content) {
                Ok(c) => c,
                Err(_) if self.fail_open => String::new(),
                Err(e) => return Err(e),
            };
            let content = if context.is_empty() {
                chunk.page_content.clone()
            } else {
                format!("{context}{}{}", self.separator, chunk.page_content)
            };
            let mut meta = chunk.metadata.clone();
            meta.insert("context".into(), context.into());
            meta.insert("original_chunk".into(), chunk.page_content.into());
            out.push(Document::with_metadata(content, meta));
        }
        Ok(out)
    }
}
