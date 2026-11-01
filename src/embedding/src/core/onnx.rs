//! Sessões do ONNX Runtime no MESMO ambiente do OCR (docling) e do chunking: um processo, um
//! ambiente, e o provedor de execução (CPU, CUDA...) escolhido do mesmo jeito (`DOCLING_RS_EP`).

use std::borrow::Cow;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use ort::session::{Session, SessionInputValue};
use ort::value::{Tensor, TensorElementType, ValueType};

use crate::error::{EmbeddingError, Result};

/// Entrada de um modelo, já com a forma.
pub enum Feed {
    Int(Vec<usize>, Vec<i64>),
    Float(Vec<usize>, Vec<f32>),
}

/// Saída em `f32`: forma e dados (por linha).
pub struct Output {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

struct Port {
    name: String,
    ty: Option<TensorElementType>,
    rank: usize,
}

pub struct OnnxModel {
    session: Mutex<Session>,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

fn port(name: &str, ty: &ValueType) -> Port {
    match ty {
        ValueType::Tensor { ty, shape, .. } => Port { name: name.to_string(), ty: Some(*ty), rank: shape.len() },
        _ => Port { name: name.to_string(), ty: None, rank: 0 },
    }
}

/// `device`: `cpu`, ou `auto`/`gpu`/`cuda`/`cuda:N` para o provedor de `DOCLING_RS_EP`.
pub fn check_device(device: &str) -> Result<bool> {
    let d = device.trim().to_ascii_lowercase();
    if d == "cpu" {
        return Ok(false);
    }
    let known = d == "auto" || d == "gpu" || d == "cuda" || d.strip_prefix("cuda:").is_some_and(|n| n.parse::<u32>().is_ok());
    if !known {
        return Err(EmbeddingError::config(format!("device must be 'cpu', 'cuda', 'cuda:N', 'gpu' or 'auto', got '{device}'")));
    }
    if d.starts_with("cuda") && !docling_onnx::compiled_providers().contains(&"cuda") {
        return Err(EmbeddingError::config("device 'cuda' requires the `cuda` feature of the embedding crate"));
    }
    Ok(true)
}

impl OnnxModel {
    pub fn load(path: &Path, device: &str) -> Result<Self> {
        let gpu = check_device(device)?;
        let builder = docling_onnx::session_builder().map_err(EmbeddingError::model)?;
        let builder = if gpu { docling_onnx::apply(builder) } else { docling_onnx::apply_host(builder) }.map_err(EmbeddingError::model)?;
        let session = docling_onnx::commit_uncached(builder, path)
            .map_err(|e| EmbeddingError::model(format!("ONNX '{}': {e}", path.display())))?;
        let inputs = session.inputs().iter().map(|o| port(o.name(), o.dtype())).collect();
        let outputs = session.outputs().iter().map(|o| port(o.name(), o.dtype())).collect();
        Ok(Self { session: Mutex::new(session), inputs, outputs })
    }

    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.iter().map(|p| p.name.as_str()).collect()
    }

    pub fn has_input(&self, name: &str) -> bool {
        self.inputs.iter().any(|p| p.name == name)
    }

    /// Posto declarado de uma entrada (0 = escalar).
    pub fn input_rank(&self, name: &str) -> Option<usize> {
        self.inputs.iter().find(|p| p.name == name).map(|p| p.rank)
    }

    /// A saída preferida entre `names`, senão a primeira com o posto pedido.
    pub fn pick_output(&self, names: &[&str], rank: usize) -> Result<String> {
        names
            .iter()
            .find_map(|n| self.outputs.iter().find(|p| p.name == *n))
            .or_else(|| self.outputs.iter().find(|p| p.rank == rank))
            .map(|p| p.name.clone())
            .ok_or_else(|| {
                let names: Vec<&str> = self.outputs.iter().map(|p| p.name.as_str()).collect();
                EmbeddingError::model(format!("ONNX model has no suitable output (rank {rank}); outputs: {}", names.join(", ")))
            })
    }

    /// Roda com as entradas nomeadas e devolve as saídas pedidas, em `f32`. Inteiros vão no tipo
    /// declarado pela entrada (`int64` ou `int32`).
    pub fn run(&self, feeds: Vec<(&str, Feed)>, outputs: &[&str]) -> Result<Vec<Output>> {
        let mut values: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(feeds.len());
        for (name, feed) in feeds {
            let ty = self.inputs.iter().find(|p| p.name == name).and_then(|p| p.ty);
            let value: SessionInputValue = match (feed, ty) {
                (Feed::Int(shape, data), Some(TensorElementType::Int32)) => {
                    Tensor::from_array((shape, data.into_iter().map(|v| v as i32).collect::<Vec<_>>())).map_err(EmbeddingError::model)?.into()
                }
                (Feed::Int(shape, data), _) => Tensor::from_array((shape, data)).map_err(EmbeddingError::model)?.into(),
                (Feed::Float(shape, data), _) => Tensor::from_array((shape, data)).map_err(EmbeddingError::model)?.into(),
            };
            values.push((Cow::Owned(name.to_string()), value));
        }
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let result = session.run(values).map_err(EmbeddingError::model)?;
        outputs
            .iter()
            .map(|name| {
                let value = result.get(*name).ok_or_else(|| EmbeddingError::model(format!("ONNX output '{name}' missing")))?;
                let (shape, data) = value.try_extract_tensor::<f32>().map_err(|e| {
                    EmbeddingError::model(format!("ONNX output '{name}': {e} (use a float32 export, not fp16/quantized output)"))
                })?;
                Ok(Output { shape: shape.iter().map(|&d| d.max(0) as usize).collect(), data: data.to_vec() })
            })
            .collect()
    }
}
