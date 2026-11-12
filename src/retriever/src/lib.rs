//! Estratégias de recuperação (retrieval) para RAG, em Rust.
//!
//! Port do `src/retrieval` do chatbot_rag (mesmos nomes, apelidos, configuração e contrato).
//! Uma estratégia por arquivo em [`strategies`]; o [`RetrievalManager`] escolhe a estratégia pela
//! configuração. Documentos são os [`chunking::Document`] do crate `chunking`, e as estratégias
//! densas usam o crate `embedding`. Integrações externas do Python (SDKs do LangChain: bancos
//! vetoriais, buscas gerenciadas, web) entram como um [`SearchProvider`] injetado.
//!
//! ```no_run
//! use retriever::{RetrievalManager, Retriever};
//! use serde_json::json;
//!
//! let manager = RetrievalManager::from_value(json!({
//!     "strategy": "hybrid",
//!     "k": 3,
//!     "embedding": {"strategy": "e5"},
//!     "documents": [
//!         {"page_content": "Fortaleza tem praias.", "metadata": {"id": 1}},
//!         {"page_content": "Rust é rápido.", "metadata": {"id": 2}},
//!     ],
//! }))?;
//! let docs = manager.retrieve("praias no Ceará")?;
//! let lexical = manager.retrieve_with("praias", json!({"strategy": "bm25", "k": 1}))?;
//! # Ok::<(), retriever::RetrievalError>(())
//! ```

#[cfg(feature = "async")]
pub mod asynch;
pub mod config;
pub mod context;
pub mod core;
pub mod deps;
pub mod error;
pub mod manager;
pub mod retriever;
pub mod strategies;

pub use chunking::Document;
pub use config::{
    EXTERNAL_PROVIDERS, FUSION_STRATEGIES, RetrievalConfig, SUPPORTED_PROVIDERS, SUPPORTED_STRATEGIES, normalize_provider,
    normalize_strategy,
};
pub use deps::{Compressor, Llm, QueryGenerator, Reranker, SearchProvider, SearchRequest, SearchType};
pub use error::{RetrievalError, Result};
pub use manager::RetrievalManager;
pub use retriever::{LEGACY_RETRIEVAL_METHODS, RETRIEVER_CONTRACT, Retriever};
