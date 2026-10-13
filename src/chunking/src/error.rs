//! Erros do crate. Configuração inválida é recusada antes do uso, e falta de capacidade
//! (modelo não carregado, feature não compilada) gera um erro explícito, nunca um no-op.

#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    #[error("configuração inválida: {0}")]
    InvalidConfig(String),
    #[error("estratégia de chunking não suportada '{0}'. Disponíveis: {1}")]
    UnsupportedStrategy(String, String),
    #[error("{0}")]
    ModelNotLoaded(String),
    #[error("a estratégia '{strategy}' precisa da feature `{feature}` do crate `chunking`")]
    FeatureDisabled { strategy: &'static str, feature: &'static str },
    #[error("modelo: {0}")]
    Model(String),
    #[error("regex: {0}")]
    Regex(String),
    #[error("JSON: {0}")]
    Json(String),
    #[error("LLM: {0}")]
    Llm(String),
    #[error("o chunker não implementa o contrato obrigatório: {0}")]
    Contract(String),
}

pub type Result<T> = std::result::Result<T, ChunkError>;

impl ChunkError {
    pub fn config(msg: impl Into<String>) -> Self {
        ChunkError::InvalidConfig(msg.into())
    }
}
