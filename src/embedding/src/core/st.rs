//! Pipeline do sentence-transformers em ONNX: lê o `modules.json` do modelo (Transformer →
//! Pooling → Dense → Normalize), os prompts do `config_sentence_transformers.json` e o
//! `max_seq_length`, e reproduz o `encode`/`encode_query`/`encode_document` do Python.
//!
//! Base das famílias densas (`sentence_transformer`, `sbert`, `bge`, `gte`, `e5`,
//! `multilingual`, `instructor`, `jina`, `nomic`).

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::config::EmbeddingConfig;
use crate::core::dense::Dense;
use crate::core::hub::ModelFiles;
use crate::core::matrix::normalize_in_place;
use crate::core::onnx::Feed;
use crate::core::options::Options;
use crate::core::pooling::Pooling;
use crate::core::text_model::{TextInput, TextModel, in_batches, max_seq_length};
use crate::embedder::InputType;
use crate::error::{EmbeddingError, Result};

/// Exports ONNX procurados, em ordem (o último é o encoder do T5, usado pelo INSTRUCTOR).
pub const TEXT_ONNX: &[&str] = &["onnx/model.onnx", "model.onnx", "onnx/encoder_model.onnx", "encoder_model.onnx"];

/// Opções de `model_options` (os kwargs do `SentenceTransformer(...)` do Python).
#[derive(Default)]
pub struct ModelOptions {
    pub revision: Option<String>,
    pub truncate_dim: Option<usize>,
    pub prompts: Option<HashMap<String, String>>,
    pub default_prompt_name: Option<String>,
}

impl ModelOptions {
    pub fn parse(map: Map<String, Value>) -> Result<Self> {
        let mut o = Options::new("model_options", &map);
        let parsed = Self {
            revision: o.string("revision")?,
            truncate_dim: o.usize("truncate_dim")?,
            prompts: o.take("prompts")?,
            default_prompt_name: o.string("default_prompt_name")?,
        };
        // Sem efeito em ONNX (não há código remoto, o dispositivo vem de `device`, o token do Hub
        // vem de HF_TOKEN e o cache de HF_HOME).
        o.ignore(&[
            "trust_remote_code",
            "token",
            "cache_folder",
            "local_files_only",
            "backend",
            "device",
            "model_kwargs",
            "tokenizer_kwargs",
            "config_kwargs",
            "similarity_fn_name",
        ]);
        o.finish()?;
        Ok(parsed)
    }
}

/// Opções de `encode_options` (os kwargs do `encode` do Python).
#[derive(Default)]
pub struct EncodeOptions {
    pub batch_size: Option<usize>,
    pub prompt: Option<String>,
    pub prompt_name: Option<String>,
    pub normalize_embeddings: bool,
    pub truncate_dim: Option<usize>,
}

impl EncodeOptions {
    pub fn parse(map: Map<String, Value>) -> Result<Self> {
        let mut o = Options::new("encode_options", &map);
        if let Some(p) = o.string("precision")?
            && p != "float32"
        {
            return Err(EmbeddingError::config(format!("encode_options.precision '{p}' is not supported (only float32)")));
        }
        if let Some(v) = o.string("output_value")?
            && v != "sentence_embedding"
        {
            return Err(EmbeddingError::config("encode_options.output_value must be 'sentence_embedding'"));
        }
        let parsed = Self {
            batch_size: o.usize("batch_size")?,
            prompt: o.string("prompt")?,
            prompt_name: o.string("prompt_name")?,
            normalize_embeddings: o.bool("normalize_embeddings", false)?,
            truncate_dim: o.usize("truncate_dim")?,
        };
        o.ignore(&["show_progress_bar", "convert_to_numpy", "convert_to_tensor", "device", "pool", "chunk_size"]);
        o.finish()?;
        Ok(parsed)
    }
}

enum Module {
    Pooling(Pooling),
    Dense(Dense),
    Normalize,
}

pub struct StModel {
    pub name: String,
    pub model: TextModel,
    output: String,
    modules: Vec<Module>,
    include_prompt: bool,
    pub prompts: HashMap<String, String>,
    pub default_prompt_name: Option<String>,
    /// `config.json` do modelo (o Jina lê as tarefas daqui).
    pub hf_config: Option<Value>,
    truncate_dim: Option<usize>,
}

fn module_kind(t: &str) -> &str {
    t.rsplit('.').next().unwrap_or(t)
}

impl StModel {
    /// Carrega o modelo `config.model_name` (local ou Hub). `options`: as da estratégia, de onde
    /// saem `model_options`, `onnx_file` e `max_seq_length`.
    pub fn load(config: &EmbeddingConfig, o: &mut Options) -> Result<Self> {
        let name = config.model_name.clone().ok_or_else(|| EmbeddingError::config(format!("model_name is required for strategy '{}'", config.strategy)))?;
        let model_options = ModelOptions::parse(o.object("model_options")?)?;
        let onnx_file = o.string("onnx_file")?;
        let max_override = o.usize("max_seq_length")?;
        let files = ModelFiles::open(&name, model_options.revision.as_deref())?;

        // modules.json; sem ele, o sentence-transformers cria Transformer + Pooling(mean).
        let modules_json = files.json("modules.json", false)?;
        let mut modules = Vec::new();
        let mut include_prompt = true;
        match modules_json.as_ref().and_then(Value::as_array) {
            None => modules.push(Module::Pooling(Pooling::default())),
            Some(list) => {
                for m in list {
                    let kind = module_kind(m.get("type").and_then(Value::as_str).unwrap_or(""));
                    let path = m.get("path").and_then(Value::as_str).unwrap_or("");
                    match kind {
                        "Transformer" | "MLMTransformer" => {}
                        "Pooling" => {
                            let cfg = files.json(&format!("{path}/config.json"), false)?.unwrap_or(Value::Null);
                            let pooling = Pooling::from_config(&cfg)?;
                            include_prompt = pooling.include_prompt;
                            modules.push(Module::Pooling(pooling));
                        }
                        "Dense" => modules.push(Module::Dense(Dense::load(&files, path)?)),
                        "Normalize" => modules.push(Module::Normalize),
                        other => {
                            return Err(EmbeddingError::model(format!("sentence-transformers module '{other}' of '{name}' is not supported")));
                        }
                    }
                }
            }
        }
        if !modules.iter().any(|m| matches!(m, Module::Pooling(_))) {
            return Err(EmbeddingError::model(format!("'{name}' has no Pooling module")));
        }
        // INSTRUCTOR: o sentence-transformers força include_prompt=false nesses modelos.
        if ["hkunlp/instructor-base", "hkunlp/instructor-large", "hkunlp/instructor-xl"].contains(&name.as_str()) {
            include_prompt = false;
            for m in modules.iter_mut() {
                if let Module::Pooling(p) = m {
                    p.include_prompt = false;
                }
            }
        }

        let st_config = files.json("config_sentence_transformers.json", false)?;
        let mut prompts: HashMap<String, String> = st_config
            .as_ref()
            .and_then(|c| c.get("prompts"))
            .and_then(Value::as_object)
            .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect())
            .unwrap_or_default();
        if let Some(p) = model_options.prompts {
            prompts = p;
        }
        let default_prompt_name = model_options
            .default_prompt_name
            .or_else(|| st_config.as_ref().and_then(|c| c.get("default_prompt_name")).and_then(Value::as_str).map(str::to_string));
        if let Some(d) = &default_prompt_name
            && !prompts.contains_key(d)
        {
            return Err(EmbeddingError::config(format!("Default prompt name '{d}' not found in the configured prompts")));
        }

        let max_length = max_override.map_or_else(|| max_seq_length(&files, 512), Ok)?;
        let model = TextModel::load(&files, TEXT_ONNX, onnx_file.as_deref(), &config.device, max_length)?;
        let output = model.onnx.pick_output(&["last_hidden_state", "token_embeddings"], 3)?;
        let hf_config = files.json("config.json", true)?;
        Ok(Self {
            name,
            model,
            output,
            modules,
            include_prompt,
            prompts,
            default_prompt_name,
            hf_config,
            truncate_dim: model_options.truncate_dim,
        })
    }

    /// Prompt efetivo, como o `encode_query`/`encode_document` + `encode` do Python.
    pub fn resolve_prompt(&self, input_type: InputType, use_task_methods: bool, enc: &EncodeOptions) -> Result<Option<String>> {
        if let Some(p) = &enc.prompt {
            return Ok(Some(p.clone()));
        }
        let mut name = enc.prompt_name.clone();
        if name.is_none() && use_task_methods {
            name = match input_type {
                InputType::Query => self.prompts.contains_key("query").then(|| "query".to_string()),
                InputType::Document => ["document", "passage", "corpus"].into_iter().find(|n| self.prompts.contains_key(*n)).map(str::to_string),
            };
        }
        match name {
            Some(n) => self
                .prompts
                .get(&n)
                .cloned()
                .map(Some)
                .ok_or_else(|| EmbeddingError::config(format!("Prompt name '{n}' not found in the configured prompts"))),
            None => Ok(self.default_prompt_name.as_ref().and_then(|d| self.prompts.get(d)).cloned()),
        }
    }

    /// Tokens do prompt, sem o especial final (`_get_prompt_length`).
    fn prompt_length(&self, prompt: &str) -> Result<usize> {
        let enc = self.model.tokenizer.encode(prompt, true).map_err(EmbeddingError::model)?;
        let mut n = enc.get_ids().len();
        if enc.get_special_tokens_mask().last() == Some(&1) {
            n = n.saturating_sub(1);
        }
        Ok(n)
    }

    /// `encode`: um vetor por entrada. `extra`: entradas adicionais do grafo para um lote de
    /// `n` linhas (ex.: `task_id`).
    pub fn encode(
        &self,
        inputs: &[TextInput],
        prompt: Option<&str>,
        batch_size: usize,
        normalize_embeddings: bool,
        truncate_dim: Option<usize>,
        extra: &dyn Fn(usize) -> Vec<(&'static str, Feed)>,
    ) -> Result<Vec<Vec<f32>>> {
        let prompted: Vec<String>;
        let inputs: Vec<TextInput> = match prompt {
            Some(p) if !p.is_empty() => {
                prompted = inputs
                    .iter()
                    .map(|i| match i {
                        TextInput::Single(t) => format!("{p}{t}"),
                        TextInput::Pair(a, _) => format!("{p}{a}"),
                    })
                    .collect();
                inputs
                    .iter()
                    .zip(&prompted)
                    .map(|(i, s)| match i {
                        TextInput::Single(_) => TextInput::Single(s),
                        TextInput::Pair(_, b) => TextInput::Pair(s, b),
                    })
                    .collect()
            }
            _ => inputs.to_vec(),
        };
        let prompt_length = match prompt {
            Some(p) if !p.is_empty() && !self.include_prompt => Some(self.prompt_length(p)?),
            _ => None,
        };
        let truncate = truncate_dim.or(self.truncate_dim);
        in_batches(&inputs, batch_size, |batch| {
            let enc = self.model.encode(batch)?;
            let out = self.model.run(&enc, extra(enc.rows), &[self.output.as_str()])?;
            let tokens = &out[0];
            let (seq, hidden) = match tokens.shape.as_slice() {
                [_, s, h] => (*s, *h),
                other => return Err(EmbeddingError::model(format!("unexpected token embeddings shape {other:?}"))),
            };
            (0..enc.rows)
                .map(|r| {
                    let rows = &tokens.data[r * seq * hidden..(r + 1) * seq * hidden];
                    let mask = &enc.mask[r * enc.seq..(r + 1) * enc.seq];
                    let mut v = Vec::new();
                    for m in &self.modules {
                        v = match m {
                            Module::Pooling(p) => p.pool(rows, seq, hidden, mask, prompt_length),
                            Module::Dense(d) => d.forward(&v)?,
                            Module::Normalize => {
                                normalize_in_place(&mut v);
                                v
                            }
                        };
                    }
                    if let Some(d) = truncate {
                        v.truncate(d);
                    }
                    if normalize_embeddings {
                        normalize_in_place(&mut v);
                    }
                    Ok(v)
                })
                .collect()
        })
    }
}

/// Sem entradas extras.
pub fn no_extra(_: usize) -> Vec<(&'static str, Feed)> {
    Vec::new()
}
