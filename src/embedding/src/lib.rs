//! Estratégias de embeddings para RAG, em Rust.
//!
//! Port do `src/embedding` do chatbot_rag (mesmos nomes, apelidos, configuração e contrato),
//! sem os provedores pagos, mais estratégias só de lógica (`bm25`, `lsa`). Uma estratégia por
//! arquivo em [`strategies`]; o [`EmbeddingManager`] escolhe a estratégia pela configuração e
//! carrega o modelo no primeiro uso. Modelos rodam em ONNX Runtime, em processo, no mesmo
//! ambiente ONNX do OCR e do chunking.
//!
//! ```no_run
//! use embedding::{Embedder, EmbeddingConfig, EmbeddingManager};
//!
//! let m = EmbeddingManager::new(EmbeddingConfig::new("e5")?)?;
//! let docs = m.embed_documents(&["Fortaleza tem praias.", "Rust é rápido."])?;
//! let query = m.embed_query("praias no Ceará")?;
//! # Ok::<(), embedding::EmbeddingError>(())
//! ```

#[cfg(feature = "async")]
pub mod asynch;
pub mod config;
pub mod core;
pub mod embedder;
pub mod error;
pub mod manager;
pub mod strategies;

pub use config::{EmbeddingConfig, FITTED_STRATEGIES, MODEL_STRATEGIES, PAID_STRATEGIES, SUPPORTED_STRATEGIES, normalize_strategy};
pub use core::matrix::cosine;
pub use embedder::{EMBEDDING_CONTRACT, Embedder, ImageInput, InputType, SparseEmbedding, as_strs};
pub use error::{EmbeddingError, Result};
pub use manager::EmbeddingManager;
pub use strategies::custom::from_fn;
