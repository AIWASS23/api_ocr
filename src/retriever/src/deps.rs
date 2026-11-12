//! Dependências injetáveis — o equivalente dos objetos que o Python recebe em
//! `provider_options` e `options` (`retriever`, `vectorstore`, `reranker`, `compressor`,
//! `query_generator`, `query_constructor`, `router`, `neighbor_loader`, `llm`).

use std::sync::Arc;

use chunking::Document;
use serde_json::{Map, Value};

use crate::error::Result;

/// Tipo de busca pedido a um provider (o `search_type` do `as_retriever` do LangChain).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchType {
    Similarity,
    Mmr,
    SimilarityScoreThreshold,
}

impl SearchType {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchType::Similarity => "similarity",
            SearchType::Mmr => "mmr",
            SearchType::SimilarityScoreThreshold => "similarity_score_threshold",
        }
    }
}

/// Pedido a um provider (o `search_kwargs` do Python).
#[derive(Clone, Debug)]
pub struct SearchRequest<'a> {
    pub query: &'a str,
    /// Vetor da consulta já calculado (multimodal / `retrieve_img`).
    pub vector: Option<&'a [f32]>,
    pub k: usize,
    /// Só no MMR.
    pub fetch_k: Option<usize>,
    pub search_type: SearchType,
    pub score_threshold: Option<f64>,
    pub filters: &'a Map<String, Value>,
    pub collection: Option<&'a str>,
}

/// Fonte de documentos externa: banco vetorial, busca gerenciada, web, grafo, SQL...
pub trait SearchProvider: Send + Sync {
    fn search(&self, request: &SearchRequest) -> Result<Vec<Document>>;

    /// Todo o acervo (`db.get_documents` do Python), para as estratégias lexicais. `None`: o
    /// provider não lista documentos.
    fn documents(&self, collection: Option<&str>) -> Result<Option<Vec<Document>>> {
        let _ = collection;
        Ok(None)
    }
}

/// Modelo de linguagem: um prompt, um texto.
pub trait Llm: Send + Sync {
    fn complete(&self, prompt: &str) -> Result<String>;
}

/// Reordena documentos para a consulta e devolve os `top_n` melhores (com
/// `metadata.retrieval_score`).
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, documents: &[Document], top_n: usize) -> Result<Vec<Document>>;
}

/// Reduz cada documento ao trecho relevante para a consulta.
pub trait Compressor: Send + Sync {
    fn compress(&self, query: &str, documents: &[Document]) -> Result<Vec<Document>>;
}

/// Gera consultas a partir da original. `mode`: `variants`, `expansion`, `decomposition`
/// (várias; `count` é o máximo) ou `rephrase`, `hyde`, `step_back` (uma; `count` é `None`).
pub trait QueryGenerator: Send + Sync {
    fn generate(&self, query: &str, mode: &str, count: Option<usize>) -> Result<Vec<String>>;
}

/// Escolhe a estratégia de uma consulta (`router`, `adaptive`).
pub type RouterFn = dyn Fn(&str, &[String]) -> String + Send + Sync;
/// Self-query: consulta reescrita e filtros extraídos dela.
pub type QueryConstructorFn = dyn Fn(&str) -> Result<(String, Map<String, Value>)> + Send + Sync;
/// Vizinhos de um trecho (`small_to_big`).
pub type NeighborLoaderFn = dyn Fn(&Document) -> Vec<Document> + Send + Sync;

/// Tudo o que pode ser injetado num `RetrievalManager`.
#[derive(Clone, Default)]
pub struct Deps {
    pub embedder: Option<Arc<dyn embedding::Embedder>>,
    pub llm: Option<Arc<dyn Llm>>,
    pub reranker: Option<Arc<dyn Reranker>>,
    pub compressor: Option<Arc<dyn Compressor>>,
    /// Provider da configuração (`provider_options.retriever`/`vectorstore` do Python).
    pub provider: Option<Arc<dyn SearchProvider>>,
    /// Componentes extras da fusão (objetos em `retrievers` no Python).
    pub components: Vec<Arc<dyn SearchProvider>>,
    /// Retriever da estratégia `custom` (`options.retriever`/`executor`).
    pub custom: Option<Arc<dyn SearchProvider>>,
    pub query_generator: Option<Arc<dyn QueryGenerator>>,
    pub query_constructor: Option<Arc<QueryConstructorFn>>,
    pub router: Option<Arc<RouterFn>>,
    pub neighbor_loader: Option<Arc<NeighborLoaderFn>>,
}

// Funções e closures servem direto como dependência (`Arc::new(|prompt: &str| ...)`).

impl<F> SearchProvider for F
where
    F: Fn(&SearchRequest) -> Result<Vec<Document>> + Send + Sync,
{
    fn search(&self, request: &SearchRequest) -> Result<Vec<Document>> {
        self(request)
    }
}

impl<F> Llm for F
where
    F: Fn(&str) -> Result<String> + Send + Sync,
{
    fn complete(&self, prompt: &str) -> Result<String> {
        self(prompt)
    }
}

impl<F> Reranker for F
where
    F: Fn(&str, &[Document], usize) -> Result<Vec<Document>> + Send + Sync,
{
    fn rerank(&self, query: &str, documents: &[Document], top_n: usize) -> Result<Vec<Document>> {
        self(query, documents, top_n)
    }
}

impl<F> Compressor for F
where
    F: Fn(&str, &[Document]) -> Result<Vec<Document>> + Send + Sync,
{
    fn compress(&self, query: &str, documents: &[Document]) -> Result<Vec<Document>> {
        self(query, documents)
    }
}

impl<F> QueryGenerator for F
where
    F: Fn(&str, &str, Option<usize>) -> Result<Vec<String>> + Send + Sync,
{
    fn generate(&self, query: &str, mode: &str, count: Option<usize>) -> Result<Vec<String>> {
        self(query, mode, count)
    }
}
