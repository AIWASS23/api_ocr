//! `jina` — Jina Embeddings v3 (`jinaai/jina-embeddings-v3`, 1024 dimensões, 8192 tokens,
//! multilíngue, Matryoshka): um adaptador LoRA por tarefa. Consultas usam a tarefa
//! `retrieval.query` e documentos `retrieval.passage`; cada tarefa tem o seu prompt (do
//! `config_sentence_transformers.json`) e o seu índice (`lora_adaptations` do `config.json`),
//! passado ao grafo ONNX na entrada `task_id`. Modelos sem `task_id` (Jina v2) usam o pipeline
//! comum.
//!
//! Diferença intencional em relação ao Python: lá o adaptador chama `encode_query`/
//! `encode_document`, que procuram prompts `query`/`document` (o v3 não tem) e não escolhem
//! tarefa; aqui vale o uso documentado pela Jina (tarefa + prompt).
//!
//! Licença do v3: CC BY-NC 4.0 (uso comercial exige licença da Jina AI).
//!
//! Opções: `task` (as duas), `query_task`, `document_task`, `use_task_prompts` (`true`) e as do
//! `sentence_transformer`.

use serde_json::Value;

use crate::config::EmbeddingConfig;
use crate::core::onnx::Feed;
use crate::core::options::Options;
use crate::core::st::{EncodeOptions, StModel};
use crate::core::text_model::TextInput;
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

pub struct JinaEmbedder {
    st: StModel,
    query_task: String,
    document_task: String,
    use_task_prompts: bool,
    /// Tarefas do `lora_adaptations`, quando o grafo recebe `task_id`.
    tasks: Option<Vec<String>>,
    task_id_rank: usize,
    encode: EncodeOptions,
    batch_size: usize,
}

pub fn load(config: &EmbeddingConfig) -> Result<JinaEmbedder> {
    let mut o = Options::new("jina", &config.options);
    let both = o.string("task")?;
    let query_task = o.string("query_task")?.or_else(|| both.clone()).unwrap_or_else(|| "retrieval.query".into());
    let document_task = o.string("document_task")?.or(both).unwrap_or_else(|| "retrieval.passage".into());
    let use_task_prompts = o.bool("use_task_prompts", true)?;
    let encode = EncodeOptions::parse(o.object("encode_options")?)?;
    let st = StModel::load(config, &mut o)?;
    o.finish()?;
    let tasks = if st.model.onnx.has_input("task_id") {
        let list: Vec<String> = st
            .hf_config
            .as_ref()
            .and_then(|c| c.get("lora_adaptations"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
            .ok_or_else(|| EmbeddingError::model("model takes task_id but config.json has no lora_adaptations"))?;
        for t in [&query_task, &document_task] {
            if !list.contains(t) {
                return Err(EmbeddingError::config(format!("jina: unknown task '{t}' (available: {})", list.join(", "))));
            }
        }
        Some(list)
    } else {
        None
    };
    let task_id_rank = st.model.onnx.input_rank("task_id").unwrap_or(0);
    let batch_size = encode.batch_size.unwrap_or(config.batch_size);
    Ok(JinaEmbedder { st, query_task, document_task, use_task_prompts, tasks, task_id_rank, encode, batch_size })
}

impl Embedder for JinaEmbedder {
    fn model_name(&self) -> &str {
        &self.st.name
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let task = match input_type {
            InputType::Query => &self.query_task,
            InputType::Document => &self.document_task,
        };
        let prompt = match (&self.encode.prompt, &self.encode.prompt_name) {
            (None, None) if self.use_task_prompts => self.st.prompts.get(task).cloned(),
            _ => self.st.resolve_prompt(input_type, false, &self.encode)?,
        };
        let task_id = self.tasks.as_ref().and_then(|l| l.iter().position(|t| t == task)).map(|i| i as i64);
        let rank = self.task_id_rank;
        let extra = move |rows: usize| -> Vec<(&'static str, Feed)> {
            match task_id {
                // Escalar (export oficial) ou um por linha.
                Some(id) if rank == 0 => vec![("task_id", Feed::Int(vec![], vec![id]))],
                Some(id) => vec![("task_id", Feed::Int(vec![rows], vec![id; rows]))],
                None => Vec::new(),
            }
        };
        let inputs: Vec<TextInput> = texts.iter().map(|t| TextInput::Single(t)).collect();
        self.st.encode(&inputs, prompt.as_deref(), self.batch_size, self.encode.normalize_embeddings, self.encode.truncate_dim, &extra)
    }
}
