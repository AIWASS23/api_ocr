//! Modelo de texto em ONNX com o tokenizador Hugging Face: tokeniza (textos ou pares), corta em
//! `max_length`, completa o lote com padding (lado e token do `tokenizer_config.json`) e monta as
//! entradas que o grafo pedir (`input_ids`, `attention_mask`, `token_type_ids`, `position_ids` e
//! extras como o `task_id` do Jina).

use std::path::Path;

use serde_json::Value;
use tokenizers::{EncodeInput, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::core::hub::ModelFiles;
use crate::core::onnx::{Feed, OnnxModel, Output};
use crate::error::{EmbeddingError, Result};

/// Um texto ou um par (o INSTRUCTOR codifica `[instrução, texto]` como par).
#[derive(Clone, Copy, Debug)]
pub enum TextInput<'a> {
    Single(&'a str),
    Pair(&'a str, &'a str),
}

impl TextInput<'_> {
    /// Tamanho usado para ordenar o lote (`_text_length` do sentence-transformers).
    pub fn len(&self) -> usize {
        match self {
            TextInput::Single(t) => t.chars().count(),
            TextInput::Pair(a, b) => a.chars().count() + b.chars().count(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Lote tokenizado: ids, máscara e tipos já com padding (`rows × seq`).
pub struct Encoded {
    pub rows: usize,
    pub seq: usize,
    pub ids: Vec<i64>,
    pub mask: Vec<i64>,
    pub types: Vec<i64>,
}

pub struct TextModel {
    pub tokenizer: Tokenizer,
    pub onnx: OnnxModel,
    pub max_length: usize,
    pad_id: u32,
    pad_left: bool,
}

/// Carrega `tokenizer.json`, sem o padding/truncamento embutidos (são aplicados aqui).
pub fn load_tokenizer(path: &Path, max_length: usize, truncate_left: bool) -> Result<(Tokenizer, Option<u32>)> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(|e| EmbeddingError::model(format!("{}: {e}", path.display())))?;
    let pad = tokenizer.get_padding().map(|p| p.pad_id);
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: if truncate_left { TruncationDirection::Left } else { TruncationDirection::Right },
        }))
        .map_err(EmbeddingError::model)?;
    Ok((tokenizer, pad))
}

/// Id do token de padding: o do `tokenizer.json`, senão o `pad_token` do `tokenizer_config.json`.
pub fn pad_id(tokenizer: &Tokenizer, embedded: Option<u32>, tokenizer_config: Option<&Value>) -> u32 {
    embedded
        .or_else(|| {
            let pad = tokenizer_config?.get("pad_token")?;
            let token = pad.as_str().or_else(|| pad.get("content").and_then(Value::as_str))?;
            tokenizer.token_to_id(token)
        })
        .unwrap_or(0)
}

impl TextModel {
    /// `max_length` já resolvido pelo chamador. `onnx`: arquivos candidatos.
    pub fn load(files: &ModelFiles, onnx: &[&str], explicit_onnx: Option<&str>, device: &str, max_length: usize) -> Result<Self> {
        let tokenizer_config = files.json("tokenizer_config.json", true)?;
        let side = |k: &str| tokenizer_config.as_ref().and_then(|c| c.get(k)).and_then(Value::as_str) == Some("left");
        let (tokenizer, embedded_pad) = load_tokenizer(&files.require("tokenizer.json")?, max_length, side("truncation_side"))?;
        let pad_id = pad_id(&tokenizer, embedded_pad, tokenizer_config.as_ref());
        let onnx = OnnxModel::load(&files.onnx(onnx, explicit_onnx)?, device)?;
        Ok(Self { tokenizer, onnx, max_length, pad_id, pad_left: side("padding_side") })
    }

    pub fn encode(&self, inputs: &[TextInput]) -> Result<Encoded> {
        let batch: Vec<EncodeInput> = inputs
            .iter()
            .map(|i| match *i {
                TextInput::Single(t) => EncodeInput::Single(t.into()),
                TextInput::Pair(a, b) => EncodeInput::Dual(a.into(), b.into()),
            })
            .collect();
        let encodings = self.tokenizer.encode_batch(batch, true).map_err(EmbeddingError::model)?;
        let seq = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0).max(1);
        let rows = encodings.len();
        let (mut ids, mut mask, mut types) = (Vec::with_capacity(rows * seq), Vec::with_capacity(rows * seq), Vec::with_capacity(rows * seq));
        for e in &encodings {
            let n = e.get_ids().len();
            let pad = seq - n;
            let fill = |v: &mut Vec<i64>, value: i64| v.extend(std::iter::repeat_n(value, pad));
            if self.pad_left {
                fill(&mut ids, self.pad_id as i64);
                fill(&mut mask, 0);
                fill(&mut types, 0);
            }
            ids.extend(e.get_ids().iter().map(|&x| x as i64));
            mask.extend(e.get_attention_mask().iter().map(|&x| x as i64));
            types.extend(e.get_type_ids().iter().map(|&x| x as i64));
            if !self.pad_left {
                fill(&mut ids, self.pad_id as i64);
                fill(&mut mask, 0);
                fill(&mut types, 0);
            }
        }
        Ok(Encoded { rows, seq, ids, mask, types })
    }

    /// Roda o grafo num lote já tokenizado. `extra`: entradas além das de texto.
    pub fn run(&self, enc: &Encoded, extra: Vec<(&str, Feed)>, outputs: &[&str]) -> Result<Vec<Output>> {
        let shape = vec![enc.rows, enc.seq];
        let mut feeds: Vec<(&str, Feed)> = Vec::new();
        for name in self.onnx.input_names() {
            let feed = match name {
                "input_ids" => Feed::Int(shape.clone(), enc.ids.clone()),
                // Nomes do BERT exportado do TensorFlow: `input_mask`, `segment_ids`.
                "attention_mask" | "input_mask" => Feed::Int(shape.clone(), enc.mask.clone()),
                "token_type_ids" | "segment_ids" => Feed::Int(shape.clone(), enc.types.clone()),
                "position_ids" => Feed::Int(shape.clone(), (0..enc.rows).flat_map(|_| 0..enc.seq as i64).collect()),
                other => match extra.iter().position(|(n, _)| *n == other) {
                    Some(_) => continue,
                    None => {
                        return Err(EmbeddingError::model(format!(
                            "ONNX model input '{other}' is not supported (inputs: {})",
                            self.onnx.input_names().join(", ")
                        )));
                    }
                },
            };
            feeds.push((name, feed));
        }
        feeds.extend(extra);
        self.onnx.run(feeds, outputs)
    }
}

/// Embeddings em lotes de `batch_size`, do texto mais longo ao mais curto (menos padding), na
/// ordem original no fim — como o `encode` do sentence-transformers.
pub fn in_batches<F>(inputs: &[TextInput], batch_size: usize, mut embed: F) -> Result<Vec<Vec<f32>>>
where
    F: FnMut(&[TextInput]) -> Result<Vec<Vec<f32>>>,
{
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(inputs[i].len()));
    let mut out: Vec<Option<Vec<f32>>> = vec![None; inputs.len()];
    for chunk in order.chunks(batch_size.max(1)) {
        let batch: Vec<TextInput> = chunk.iter().map(|&i| inputs[i]).collect();
        let vectors = embed(&batch)?;
        if vectors.len() != chunk.len() {
            return Err(EmbeddingError::model("model returned a different number of vectors than inputs"));
        }
        for (&i, v) in chunk.iter().zip(vectors) {
            out[i] = Some(v);
        }
    }
    Ok(out.into_iter().map(Option::unwrap_or_default).collect())
}

/// Tamanho máximo de sequência como o sentence-transformers: `sentence_bert_config.json`, senão
/// o menor entre `max_position_embeddings` e o `model_max_length` do tokenizador.
pub fn max_seq_length(files: &ModelFiles, fallback: usize) -> Result<usize> {
    if let Some(n) = files.json("sentence_bert_config.json", false)?.and_then(|c| c.get("max_seq_length").and_then(Value::as_u64)) {
        return Ok(n as usize);
    }
    let model = files.json("config.json", true)?.and_then(|c| c.get("max_position_embeddings").and_then(Value::as_u64));
    let tokenizer = files
        .json("tokenizer_config.json", true)?
        .and_then(|c| c.get("model_max_length").and_then(Value::as_f64))
        .filter(|&n| n > 0.0 && n < 1e9)
        .map(|n| n as u64);
    Ok(match (model, tokenizer) {
        (Some(a), Some(b)) => a.min(b) as usize,
        (Some(a), None) | (None, Some(a)) => a as usize,
        (None, None) => fallback,
    })
}
