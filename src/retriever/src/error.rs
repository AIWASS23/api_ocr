use thiserror::Error;

#[derive(Debug, Error)]
pub enum RetrievalError {
    /// Configuração inválida (o `ValueError` do Python).
    #[error("{0}")]
    InvalidConfig(String),
    #[error("Unsupported retrieval strategy '{0}'. Available: {1}")]
    UnsupportedStrategy(String, String),
    #[error("Unsupported retriever provider '{0}'. Available: {1}")]
    UnsupportedProvider(String, String),
    /// Integração externa do Python (SDK do LangChain): fora deste crate.
    #[error(
        "provider '{0}' is an external integration (a LangChain SDK in Python) and is not included; \
         inject a SearchProvider with RetrievalManager::with_provider"
    )]
    ExternalProvider(String),
    /// Falta uma dependência injetável (embedder, LLM, provider, roteador...).
    #[error("{0}")]
    MissingDependency(String),
    #[error("strategy '{strategy}' requires the `{feature}` feature of the retriever crate")]
    FeatureDisabled { strategy: &'static str, feature: &'static str },
    /// Entrada inválida (consulta vazia, documentos sem vocabulário).
    #[error("{0}")]
    Input(String),
    #[error("embedding: {0}")]
    Embedding(#[from] embedding::EmbeddingError),
    /// Modelo de reranking ou resposta de LLM.
    #[error("{0}")]
    Model(String),
    /// Todas as estratégias do `fallback` falharam.
    #[error("all retrieval strategies failed: {0}")]
    AllFailed(String),
}

impl RetrievalError {
    pub fn config(msg: impl Into<String>) -> Self {
        RetrievalError::InvalidConfig(msg.into())
    }

    pub fn missing(msg: impl Into<String>) -> Self {
        RetrievalError::MissingDependency(msg.into())
    }

    pub fn model(msg: impl std::fmt::Display) -> Self {
        RetrievalError::Model(msg.to_string())
    }
}

pub type Result<T> = std::result::Result<T, RetrievalError>;
