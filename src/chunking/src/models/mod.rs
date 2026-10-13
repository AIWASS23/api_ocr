//! Modelos e motores usados por algumas estratégias. Cada um é um trait, com implementação em
//! Rust atrás de uma feature, para que possa ser trocado ou simulado nos testes.

pub mod embeddings;
pub mod llm;
pub mod punkt;
pub mod spacy;
pub mod token_embeddings;
pub mod tokenizer;
