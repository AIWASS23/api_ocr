//! `instructor` — INSTRUCTOR (`hkunlp/instructor-base/large/xl`): cada texto vai ao modelo como o
//! par `[instrução, texto]`, com uma instrução para consultas e outra para documentos (o
//! `_prepare` do adaptador Python). Pipeline: encoder T5 → média → `Dense` → normalização.
//!
//! Os repositórios `hkunlp/*` não publicam ONNX: o encoder vem do espelho `Xenova/instructor-*`
//! (`onnx/encoder_model.onnx`) e a camada `Dense` (só em `pytorch_model.bin`) do original.
//!
//! Padrão: `hkunlp/instructor-large` (768 dimensões). Opções: `query_instruction`
//! (`"Represent the query for retrieving supporting documents:"`), `document_instruction`
//! (`"Represent the document for retrieval:"`) e as do `sentence_transformer` (menos os
//! prefixos).

use crate::config::EmbeddingConfig;
use crate::core::options::Options;
use crate::core::st::{EncodeOptions, StModel, no_extra};
use crate::core::text_model::TextInput;
use crate::embedder::{Embedder, InputType};
use crate::error::Result;

pub const DEFAULT_QUERY_INSTRUCTION: &str = "Represent the query for retrieving supporting documents:";
pub const DEFAULT_DOCUMENT_INSTRUCTION: &str = "Represent the document for retrieval:";

pub struct InstructorEmbedder {
    st: StModel,
    query_instruction: String,
    document_instruction: String,
    use_task_methods: bool,
    encode: EncodeOptions,
    batch_size: usize,
}

pub fn load(config: &EmbeddingConfig) -> Result<InstructorEmbedder> {
    let mut o = Options::new("instructor", &config.options);
    let query_instruction = o.string("query_instruction")?.unwrap_or_else(|| DEFAULT_QUERY_INSTRUCTION.into());
    let document_instruction = o.string("document_instruction")?.unwrap_or_else(|| DEFAULT_DOCUMENT_INSTRUCTION.into());
    let use_task_methods = o.bool("use_task_methods", true)?;
    let encode = EncodeOptions::parse(o.object("encode_options")?)?;
    let st = StModel::load(config, &mut o)?;
    o.finish()?;
    let batch_size = encode.batch_size.unwrap_or(config.batch_size);
    Ok(InstructorEmbedder { st, query_instruction, document_instruction, use_task_methods, encode, batch_size })
}

impl Embedder for InstructorEmbedder {
    fn model_name(&self) -> &str {
        &self.st.name
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let instruction = match input_type {
            InputType::Query => &self.query_instruction,
            InputType::Document => &self.document_instruction,
        };
        let inputs: Vec<TextInput> = texts.iter().map(|t| TextInput::Pair(instruction, t)).collect();
        let prompt = self.st.resolve_prompt(input_type, self.use_task_methods, &self.encode)?;
        self.st.encode(&inputs, prompt.as_deref(), self.batch_size, self.encode.normalize_embeddings, self.encode.truncate_dim, &no_extra)
    }
}
