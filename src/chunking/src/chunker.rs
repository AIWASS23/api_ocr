//! Contrato obrigatório de toda estratégia — o equivalente de `ChunkBase`/`CHUNK_CONTRACT`.
//!
//! | Python (`ChunkBase`)        | Rust (`Chunker`)                         |
//! |-----------------------------|------------------------------------------|
//! | `chunk()` / `split_text()`  | [`Chunker::chunk`] / [`Chunker::split_text`] |
//! | `create_documents()`        | [`Chunker::create_documents`]            |
//! | `split_documents()`         | [`Chunker::split_documents`]             |
//! | `transform_documents()`     | [`Chunker::transform_documents`]         |
//! | `batch()`                   | [`Chunker::batch`]                       |
//! | `achunk()`/`abatch()`/`atransform_documents()` | [`crate::asynch`] (feature `async`) |
//!
//! Extensão em relação ao Python: [`Chunker::chunk_documents`] devolve cada chunk com os
//! metadados próprios da estratégia (cabeçalhos, artigo de lei, id do pai, embedding...). Nos
//! métodos de documento, esses metadados são somados aos do documento de origem.

use crate::document::{Document, Metadata};
use crate::error::{ChunkError, Result};

pub trait Chunker: Send + Sync {
    /// Nome canônico da estratégia.
    fn name(&self) -> &str;
    fn chunk_size(&self) -> usize;
    fn chunk_overlap(&self) -> usize;

    /// Divide um texto em chunks já normalizados (sem espaços nas pontas, sem vazios).
    fn chunk(&self, text: &str) -> Result<Vec<String>>;

    /// Chunks com os metadados próprios da estratégia. Padrão: sem metadados extras.
    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        Ok(self.chunk(text)?.into_iter().map(Document::new).collect())
    }

    /// Alias compatível com o LangChain.
    fn split_text(&self, text: &str) -> Result<Vec<String>> {
        self.chunk(text)
    }

    /// Divide textos preservando os metadados correspondentes.
    fn create_documents(&self, texts: &[&str], metadatas: Option<&[Metadata]>) -> Result<Vec<Document>> {
        if let Some(m) = metadatas
            && m.len() != texts.len()
        {
            return Err(ChunkError::config("metadatas and texts must have the same length"));
        }
        let mut out = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let base = metadatas.map(|m| m[i].clone()).unwrap_or_default();
            for chunk in self.chunk_documents(text)? {
                let mut metadata = base.clone();
                metadata.extend(chunk.metadata);
                out.push(Document::with_metadata(chunk.page_content, metadata));
            }
        }
        Ok(out)
    }

    /// Divide documentos sem descartar os metadados deles.
    fn split_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        let texts: Vec<&str> = documents.iter().map(|d| d.page_content.as_str()).collect();
        let metadatas: Vec<Metadata> = documents.iter().map(|d| d.metadata.clone()).collect();
        self.create_documents(&texts, Some(&metadatas))
    }

    fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        self.split_documents(documents)
    }

    /// Vários textos, preservando a ordem.
    fn batch(&self, texts: &[&str]) -> Result<Vec<Vec<String>>> {
        texts.iter().map(|t| self.chunk(t)).collect()
    }
}

/// Tamanho e sobreposição validados como no `ChunkBase` do Python:
/// `chunk_size >= 1` e `0 <= chunk_overlap < chunk_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizing {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Sizing {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Result<Self> {
        if chunk_size < 1 {
            return Err(ChunkError::config("chunk_size must be a positive integer"));
        }
        if chunk_overlap >= chunk_size {
            return Err(ChunkError::config("chunk_overlap must be an integer between 0 and chunk_size - 1"));
        }
        Ok(Self { chunk_size, chunk_overlap })
    }
}

/// Implementa os acessores comuns a partir de um campo `sizing: Sizing` e um nome fixo.
#[macro_export]
macro_rules! sizing_accessors {
    ($name:expr) => {
        fn name(&self) -> &str {
            $name
        }
        fn chunk_size(&self) -> usize {
            self.sizing.chunk_size
        }
        fn chunk_overlap(&self) -> usize {
            self.sizing.chunk_overlap
        }
    };
}
