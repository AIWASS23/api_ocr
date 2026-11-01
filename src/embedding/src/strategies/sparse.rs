//! `sparse` — SPLADE (`SparseEncoder` do sentence-transformers, `SparseEncoderAdapter` do
//! chatbot): o modelo de linguagem mascarado dá um logit por token do vocabulário em cada
//! posição; o vetor do texto é o máximo de `log(1 + relu(logit))` sobre as posições. Um vetor
//! do tamanho do vocabulário (30522 no BERT), quase todo zero e interpretável (cada posição é um
//! termo, com expansão para termos relacionados).
//!
//! Padrão: `prithivida/Splade_PP_en_v1` (Apache-2.0, com ONNX). O padrão do Python,
//! `naver/splade-v3`, é restrito (licença não comercial, acesso por aprovação) e não tem ONNX;
//! use-o com um export local, se tiver a licença. Prefira [`Embedder::embed_sparse`] para obter
//! só os termos ativos.
//!
//! Opções: `max_active_dims` (só os N maiores pesos de cada vetor), `onnx_file`,
//! `max_seq_length`, `model_options` (`revision`), `use_task_methods` (prompts `query`/`document`
//! do modelo, quando houver).

use serde_json::Value;

use crate::config::EmbeddingConfig;
use crate::core::hub::ModelFiles;
use crate::core::options::Options;
use crate::core::st::TEXT_ONNX;
use crate::core::text_model::{Encoded, TextInput, TextModel, max_seq_length};
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

/// Teto de memória do tensor de logits por execução (linhas × posições × vocabulário).
const LOGITS_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pool {
    Max,
    Sum,
}

pub struct SparseEmbedder {
    name: String,
    model: TextModel,
    output: String,
    vocab: usize,
    pooling: Pool,
    query_prompt: Option<String>,
    document_prompt: Option<String>,
    max_active_dims: Option<usize>,
    batch_size: usize,
}

pub fn load(config: &EmbeddingConfig) -> Result<SparseEmbedder> {
    let mut o = Options::new("sparse", &config.options);
    let max_active_dims = o.usize("max_active_dims")?;
    let onnx_file = o.string("onnx_file")?;
    let max_override = o.usize("max_seq_length")?;
    let use_task_methods = o.bool("use_task_methods", true)?;
    let mut mo = Options::new("model_options", &o.object("model_options")?);
    let revision = mo.string("revision")?;
    mo.ignore(&["trust_remote_code", "token", "cache_folder", "local_files_only", "backend", "device", "model_kwargs"]);
    mo.finish()?;
    o.finish()?;
    let name = config.model_name.clone().ok_or_else(|| EmbeddingError::config("model_name is required for sparse embeddings"))?;
    let files = ModelFiles::open(&name, revision.as_deref())?;

    // 1_SpladePooling/config.json (ou o padrão: máximo de log1p(relu)).
    let modules = files.json("modules.json", false)?;
    let pooling_dir = modules
        .as_ref()
        .and_then(Value::as_array)
        .and_then(|l| l.iter().find(|m| m.get("type").and_then(Value::as_str).is_some_and(|t| t.ends_with("SpladePooling"))))
        .and_then(|m| m.get("path").and_then(Value::as_str))
        .map(str::to_string);
    let pooling_cfg = match &pooling_dir {
        Some(dir) => files.json(&format!("{dir}/config.json"), false)?,
        None => None,
    };
    let pooling = match pooling_cfg.as_ref().and_then(|c| c.get("pooling_strategy")).and_then(Value::as_str) {
        None | Some("max") => Pool::Max,
        Some("sum") => Pool::Sum,
        Some(other) => return Err(EmbeddingError::model(format!("SPLADE pooling '{other}' is not supported"))),
    };
    if let Some(act) = pooling_cfg.as_ref().and_then(|c| c.get("activation_function")).and_then(Value::as_str)
        && act != "relu"
    {
        return Err(EmbeddingError::model(format!("SPLADE activation '{act}' is not supported")));
    }

    let st_config = files.json("config_sentence_transformers.json", false)?;
    let prompt = |k: &str| {
        st_config.as_ref().and_then(|c| c.get("prompts")).and_then(|p| p.get(k)).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string)
    };
    let (query_prompt, document_prompt) =
        if use_task_methods { (prompt("query"), ["document", "passage", "corpus"].iter().find_map(|k| prompt(k))) } else { (None, None) };

    let max_length = max_override.map_or_else(|| max_seq_length(&files, 512), Ok)?;
    let model = TextModel::load(&files, TEXT_ONNX, onnx_file.as_deref(), &config.device, max_length)?;
    let output = model.onnx.pick_output(&["logits", "last_hidden_state", "token_embeddings"], 3)?;
    let vocab = files
        .json("config.json", true)?
        .and_then(|c| c.get("vocab_size").and_then(Value::as_u64))
        .map_or_else(|| model.tokenizer.get_vocab_size(true), |v| v as usize);
    Ok(SparseEmbedder { name, model, output, vocab, pooling, query_prompt, document_prompt, max_active_dims, batch_size: config.batch_size })
}

/// Linhas `[start, end)` de um lote já tokenizado.
fn slice(enc: &Encoded, start: usize, end: usize) -> Encoded {
    let s = enc.seq;
    Encoded {
        rows: end - start,
        seq: s,
        ids: enc.ids[start * s..end * s].to_vec(),
        mask: enc.mask[start * s..end * s].to_vec(),
        types: enc.types[start * s..end * s].to_vec(),
    }
}

impl SparseEmbedder {
    fn pool(&self, logits: &[f32], seq: usize, vocab: usize, mask: &[i64]) -> Vec<f32> {
        let mut v = vec![0f32; vocab];
        for t in (0..seq).filter(|&t| mask[t] != 0) {
            for (a, &x) in v.iter_mut().zip(&logits[t * vocab..(t + 1) * vocab]) {
                let w = x.max(0.0).ln_1p();
                match self.pooling {
                    Pool::Max => *a = a.max(w),
                    Pool::Sum => *a += w,
                }
            }
        }
        if let Some(k) = self.max_active_dims
            && k < vocab
        {
            let mut sorted: Vec<f32> = v.iter().copied().filter(|&x| x > 0.0).collect();
            if sorted.len() > k {
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let threshold = sorted[k - 1];
                let mut kept = 0;
                for x in v.iter_mut() {
                    if *x >= threshold && *x > 0.0 && kept < k {
                        kept += 1;
                    } else {
                        *x = 0.0;
                    }
                }
            }
        }
        v
    }
}

impl Embedder for SparseEmbedder {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let prompt = match input_type {
            InputType::Query => self.query_prompt.as_deref(),
            InputType::Document => self.document_prompt.as_deref(),
        }
        .unwrap_or("");
        let prepared: Vec<String> = texts.iter().map(|t| format!("{prompt}{t}")).collect();
        let mut out = Vec::with_capacity(texts.len());
        for batch in prepared.chunks(self.batch_size.max(1)) {
            let inputs: Vec<TextInput> = batch.iter().map(|t| TextInput::Single(t)).collect();
            let enc = self.model.encode(&inputs)?;
            let per_run = (LOGITS_BUDGET_BYTES / (enc.seq * self.vocab * 4).max(1)).max(1);
            let mut start = 0;
            while start < enc.rows {
                let end = (start + per_run).min(enc.rows);
                let part = slice(&enc, start, end);
                let res = self.model.run(&part, Vec::new(), &[self.output.as_str()])?;
                let [_, seq, vocab] = res[0].shape.as_slice() else {
                    return Err(EmbeddingError::model(format!("unexpected MLM logits shape {:?}", res[0].shape)));
                };
                for r in 0..part.rows {
                    let logits = &res[0].data[r * seq * vocab..(r + 1) * seq * vocab];
                    out.push(self.pool(logits, *seq, *vocab, &part.mask[r * part.seq..(r + 1) * part.seq]));
                }
                start = end;
            }
        }
        Ok(out)
    }
}
