//! Estratégias de chunking para RAG, em Rust.
//!
//! Port das 25 estratégias do `src/chunking` do chatbot_rag (mesmos nomes, apelidos,
//! configuração e resultados) mais 8 estratégias estruturais/contextuais. Uma estratégia por
//! arquivo em [`strategies`]; o [`ChunkManager`] escolhe a estratégia pela configuração.
//!
//! ```no_run
//! use chunking::{ChunkConfig, ChunkManager, Chunker};
//!
//! let manager = ChunkManager::new(ChunkConfig::new("sentence", 800, 80)?)?;
//! let chunks = manager.chunk("Primeira frase. Segunda frase.")?;
//! # Ok::<(), chunking::ChunkError>(())
//! ```

pub mod adapter;
#[cfg(feature = "async")]
pub mod asynch;
pub mod chunker;
pub mod config;
pub mod core;
pub mod document;
pub mod error;
pub mod manager;
pub mod models;
pub mod strategies;

pub use adapter::ChunkerAdapter;
pub use chunker::{Chunker, Sizing};
pub use manager::ChunkManager;
pub use config::{ChunkConfig, MODEL_STRATEGIES, SUPPORTED_STRATEGIES, normalize_strategy, requires_model};
pub use document::{Document, Metadata};
pub use error::{ChunkError, Result};
