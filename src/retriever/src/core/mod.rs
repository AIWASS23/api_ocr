//! Peças comuns às estratégias.

pub mod documents;
pub mod filters;
pub mod rank;
pub mod text;
pub mod time;
pub mod vectors;

#[cfg(feature = "onnx")]
pub mod cross_encoder;
#[cfg(feature = "llm")]
pub mod llm;
