use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbeddingError {
    /// Configuração inválida (o `ValueError`/`TypeError` do Python).
    #[error("{0}")]
    InvalidConfig(String),
    #[error("Unsupported embedding strategy '{0}'. Available: {1}")]
    UnsupportedStrategy(String, String),
    /// Provedor pago (OpenAI, Azure, Cohere, Voyage, Gemini, Mistral, HF Inference): fora deste crate.
    #[error("embedding strategy '{0}' is a paid provider and is not included; use a local strategy")]
    PaidProvider(String),
    #[error("strategy '{strategy}' requires the `{feature}` feature of the embedding crate")]
    FeatureDisabled { strategy: &'static str, feature: &'static str },
    /// Falha ao carregar ou rodar um modelo (arquivos, ONNX Runtime, tokenizador).
    #[error("{0}")]
    Model(String),
    /// Entrada inválida (textos, imagens, ordem de uso como consulta antes do ajuste).
    #[error("{0}")]
    Input(String),
    #[error("strategy '{0}' does not support image embeddings")]
    ImagesUnsupported(String),
    /// Resposta de um servidor de embeddings (Ollama).
    #[error("{0}")]
    Http(String),
}

impl EmbeddingError {
    pub fn config(msg: impl Into<String>) -> Self {
        EmbeddingError::InvalidConfig(msg.into())
    }

    pub fn model(msg: impl std::fmt::Display) -> Self {
        EmbeddingError::Model(msg.to_string())
    }

    pub fn input(msg: impl Into<String>) -> Self {
        EmbeddingError::Input(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;
