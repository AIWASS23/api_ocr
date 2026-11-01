//! Uma estratégia por arquivo. As do Python que não são de provedores pagos, na mesma ordem, e
//! as novas só de lógica (`bm25`, `lsa`). As famílias do sentence-transformers (`sbert`, `bge`,
//! `gte`, `e5`, `multilingual`, `nomic`) compartilham o pipeline de `sentence_transformer.rs`.

#[cfg(feature = "onnx")]
pub mod bge;
pub mod bm25;
#[cfg(feature = "vision")]
pub mod clip;
pub mod custom;
#[cfg(feature = "onnx")]
pub mod e5;
#[cfg(feature = "onnx")]
pub mod gte;
pub mod hashing;
#[cfg(feature = "onnx")]
pub mod huggingface;
#[cfg(feature = "onnx")]
pub mod instructor;
#[cfg(feature = "onnx")]
pub mod jina;
pub mod lsa;
#[cfg(feature = "onnx")]
pub mod multilingual;
#[cfg(feature = "onnx")]
pub mod nomic;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "onnx")]
pub mod sbert;
#[cfg(feature = "onnx")]
pub mod sentence_transformer;
#[cfg(feature = "vision")]
pub mod siglip;
#[cfg(feature = "onnx")]
pub mod sparse;
pub mod tfidf;
