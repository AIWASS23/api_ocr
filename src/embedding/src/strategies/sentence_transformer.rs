//! `sentence_transformer` — qualquer modelo do sentence-transformers (`SentenceTransformerAdapter`
//! do chatbot): pipeline do `modules.json` (Transformer → Pooling → Dense → Normalize), prompts
//! de consulta/documento do próprio modelo (`encode_query`/`encode_document`) e, nas famílias que
//! pedem, prefixos (`query: `/`passage: ` do E5, `search_query: ` do Nomic).
//!
//! Padrão: `sentence-transformers/all-MiniLM-L6-v2`. Roda em ONNX Runtime (o `onnx/model.onnx`
//! do repositório ou de um espelho), em processo.
//!
//! Opções: `query_prefix`, `document_prefix`, `use_task_methods` (`true`: prompts `query` e
//! `document`/`passage`/`corpus` do modelo), `model_options` (`revision`, `truncate_dim`,
//! `prompts`, `default_prompt_name`), `encode_options` (`batch_size`, `prompt`, `prompt_name`,
//! `normalize_embeddings`, `truncate_dim`), `onnx_file` (ex.: `onnx/model_quantized.onnx`),
//! `max_seq_length`.

use crate::config::EmbeddingConfig;
use crate::core::options::Options;
use crate::core::st::{EncodeOptions, StModel, no_extra};
use crate::core::text_model::TextInput;
use crate::embedder::{Embedder, InputType};
use crate::error::Result;

/// Prefixos padrão de uma família (o `_prepare` do adaptador Python).
#[derive(Clone, Copy, Debug, Default)]
pub struct Prefixes {
    pub query: &'static str,
    pub document: &'static str,
}

pub struct SentenceTransformerEmbedder {
    st: StModel,
    query_prefix: String,
    document_prefix: String,
    use_task_methods: bool,
    encode: EncodeOptions,
    batch_size: usize,
}

impl SentenceTransformerEmbedder {
    /// Carrega um modelo da família `strategy` com os prefixos padrão dela.
    pub fn load_family(config: &EmbeddingConfig, strategy: &'static str, prefixes: Prefixes) -> Result<Self> {
        let mut o = Options::new(strategy, &config.options);
        let query_prefix = o.string("query_prefix")?.unwrap_or_else(|| prefixes.query.to_string());
        let document_prefix = o.string("document_prefix")?.unwrap_or_else(|| prefixes.document.to_string());
        let use_task_methods = o.bool("use_task_methods", true)?;
        let encode = EncodeOptions::parse(o.object("encode_options")?)?;
        let st = StModel::load(config, &mut o)?;
        o.finish()?;
        let batch_size = encode.batch_size.unwrap_or(config.batch_size);
        Ok(Self { st, query_prefix, document_prefix, use_task_methods, encode, batch_size })
    }

    pub fn st(&self) -> &StModel {
        &self.st
    }
}

/// `sentence_transformer`: sem prefixos.
pub fn load(config: &EmbeddingConfig) -> Result<SentenceTransformerEmbedder> {
    SentenceTransformerEmbedder::load_family(config, "sentence_transformer", Prefixes::default())
}

impl Embedder for SentenceTransformerEmbedder {
    fn model_name(&self) -> &str {
        &self.st.name
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let prefix = match input_type {
            InputType::Query => &self.query_prefix,
            InputType::Document => &self.document_prefix,
        };
        let prepared: Vec<String> = texts.iter().map(|t| format!("{prefix}{t}")).collect();
        let inputs: Vec<TextInput> = prepared.iter().map(|t| TextInput::Single(t)).collect();
        let prompt = self.st.resolve_prompt(input_type, self.use_task_methods, &self.encode)?;
        self.st.encode(&inputs, prompt.as_deref(), self.batch_size, self.encode.normalize_embeddings, self.encode.truncate_dim, &no_extra)
    }
}
