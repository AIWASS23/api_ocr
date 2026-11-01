//! Peças comuns às estratégias: opções, pós-processamento, analisador de texto do scikit-learn,
//! vetorizadores, SVD e, com a feature `onnx`, arquivos de modelo, ONNX Runtime, pooling e o
//! pipeline do sentence-transformers.

pub mod analyzer;
pub mod matrix;
pub mod murmur;
pub mod options;
pub mod pooling;
pub mod stop_words;
pub mod svd;
pub mod vectorizer;

#[cfg(feature = "onnx")]
pub mod dense;
#[cfg(feature = "onnx")]
pub mod hub;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(feature = "onnx")]
pub mod st;
#[cfg(feature = "onnx")]
pub mod text_model;
#[cfg(feature = "onnx")]
pub mod torch_pickle;
#[cfg(feature = "vision")]
pub mod vision;
