//! Contrato comum a todas as estratégias — o equivalente de `embedding_model_base.py`.
//!
//! Toda estratégia implementa [`Embedder::embed`] (textos de documento ou de consulta) e, se for
//! multimodal, [`Embedder::embed_images`]. O resto do contrato do Python (`embed_docs`,
//! `embed_documents`, `embed_query`, `embed_images`) vem pronto, e as versões assíncronas
//! (`aembed`...) ficam em `asynch` (feature `async`).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{EmbeddingError, Result};

/// Os métodos do contrato do Python (`EMBEDDING_CONTRACT`).
pub const EMBEDDING_CONTRACT: &[&str] = &[
    "embed",
    "embed_docs",
    "embed_documents",
    "embed_query",
    "embed_images",
    "aembed",
    "aembed_documents",
    "aembed_query",
    "aembed_images",
];

/// Documento (o que vai para o índice) ou consulta. Vários modelos codificam os dois de jeitos
/// diferentes (prefixos do E5/Nomic, instruções do INSTRUCTOR, tarefas do Jina, BM25).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    #[default]
    Document,
    Query,
}

impl InputType {
    pub fn as_str(self) -> &'static str {
        match self {
            InputType::Document => "document",
            InputType::Query => "query",
        }
    }
}

impl std::str::FromStr for InputType {
    type Err = EmbeddingError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "document" => Ok(InputType::Document),
            "query" => Ok(InputType::Query),
            _ => Err(EmbeddingError::input("input_type must be 'document' or 'query'")),
        }
    }
}

/// Imagem para os modelos multimodais.
#[derive(Clone, Debug)]
pub enum ImageInput {
    Path(PathBuf),
    /// Arquivo codificado (PNG, JPEG, WebP, BMP, TIFF, GIF).
    Bytes(Vec<u8>),
    #[cfg(feature = "vision")]
    Image(image::DynamicImage),
}

impl From<PathBuf> for ImageInput {
    fn from(p: PathBuf) -> Self {
        ImageInput::Path(p)
    }
}

impl From<&str> for ImageInput {
    fn from(p: &str) -> Self {
        ImageInput::Path(PathBuf::from(p))
    }
}

impl From<Vec<u8>> for ImageInput {
    fn from(b: Vec<u8>) -> Self {
        ImageInput::Bytes(b)
    }
}

/// Vetor esparso (`índices`, `valores`), ordenado por índice. É o formato dos vetores esparsos
/// do Qdrant, Milvus e afins.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SparseEmbedding {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseEmbedding {
    /// Só as posições diferentes de zero de um vetor denso.
    pub fn from_dense(dense: &[f32]) -> Self {
        let mut s = SparseEmbedding::default();
        for (i, &v) in dense.iter().enumerate() {
            if v != 0.0 {
                s.indices.push(i as u32);
                s.values.push(v);
            }
        }
        s
    }

    pub fn to_dense(&self, dimension: usize) -> Vec<f32> {
        let mut out = vec![0.0; dimension];
        for (&i, &v) in self.indices.iter().zip(&self.values) {
            if let Some(slot) = out.get_mut(i as usize) {
                *slot = v;
            }
        }
        out
    }
}

pub trait Embedder: Send + Sync {
    /// Nome do modelo (ou da estratégia, quando ela não usa modelo).
    fn model_name(&self) -> &str;

    /// Um vetor por texto, na mesma ordem.
    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>>;

    /// Se [`Embedder::embed_images`] é suportado.
    fn supports_images(&self) -> bool {
        false
    }

    /// Um vetor por imagem (modelos multimodais).
    fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>> {
        let _ = images;
        Err(EmbeddingError::ImagesUnsupported(self.model_name().to_string()))
    }

    /// Alias de compatibilidade com os modelos originais (`embed_docs`).
    fn embed_docs(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(documents, InputType::Document)
    }

    /// Contrato do LangChain para vector stores.
    fn embed_documents(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(documents, InputType::Document)
    }

    /// Uma consulta, no modo de consulta do modelo quando ele tem um.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.embed(&[query], InputType::Query)?.into_iter().next().ok_or_else(|| EmbeddingError::model("o embedder não devolveu vetor"))
    }

    /// Vetores esparsos (só as posições não nulas). Útil para `sparse`, `tfidf`, `hashing` e
    /// `bm25`, cujos vetores densos têm o tamanho do vocabulário.
    fn embed_sparse(&self, texts: &[&str], input_type: InputType) -> Result<Vec<SparseEmbedding>> {
        Ok(self.embed(texts, input_type)?.iter().map(|v| SparseEmbedding::from_dense(v)).collect())
    }
}

/// Owned → emprestado, para chamar o contrato com `Vec<String>`.
pub fn as_strs(texts: &[String]) -> Vec<&str> {
    texts.iter().map(String::as_str).collect()
}
