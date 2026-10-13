#[derive(Debug, thiserror::Error)]
pub enum PrepError {
    #[error("falha ao converter: {0}")]
    Conversion(#[from] docling::ConversionError),
    #[error("erro HTTP: {0}")]
    Http(#[from] reqwest::Error),
    #[error("erro no VLM: {0}")]
    Vlm(String),
    #[error("pânico durante a conversão")]
    Panic,
    #[error("formato inválido: '{0}'. Use um destes: ['txt', 'markdown']")]
    InvalidFormat(String),
}

/// Erro por arquivo, como o `{"file": ..., "error": ...}` do Python.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileError {
    pub file: String,
    pub error: String,
}
