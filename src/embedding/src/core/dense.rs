//! O módulo `Dense` do sentence-transformers (`y = act(W·x + b)`), com pesos de
//! `model.safetensors` ou `pytorch_model.bin`.

use std::path::Path;

use serde_json::Value;

use crate::core::hub::ModelFiles;
use crate::core::torch_pickle;
use crate::error::{EmbeddingError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Tanh,
    Relu,
    Sigmoid,
}

impl Activation {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name.rsplit('.').next().unwrap_or(name) {
            "Identity" => Activation::Identity,
            "Tanh" => Activation::Tanh,
            "ReLU" => Activation::Relu,
            "Sigmoid" => Activation::Sigmoid,
            other => return Err(EmbeddingError::model(format!("Dense activation '{other}' is not supported"))),
        })
    }

    fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Identity => x,
            Activation::Tanh => x.tanh(),
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }
}

pub struct Dense {
    in_features: usize,
    out_features: usize,
    /// `out × in`, por linha (o `nn.Linear.weight`).
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    activation: Activation,
}

fn safetensor(path: &Path, name: &str) -> Result<Option<(Vec<usize>, Vec<f32>)>> {
    let bytes = std::fs::read(path).map_err(|e| EmbeddingError::model(format!("{}: {e}", path.display())))?;
    let st = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| EmbeddingError::model(format!("{}: {e}", path.display())))?;
    let Ok(view) = st.tensor(name) else { return Ok(None) };
    let data = view.data();
    let values: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        safetensors::Dtype::BF16 => data.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)).collect(),
        other => return Err(EmbeddingError::model(format!("Dense weights in {other:?} are not supported"))),
    };
    Ok(Some((view.shape().to_vec(), values)))
}

impl Dense {
    /// `dir`: pasta do módulo (`2_Dense`).
    pub fn load(files: &ModelFiles, dir: &str) -> Result<Self> {
        let config = files.json(&format!("{dir}/config.json"), false)?.ok_or_else(|| EmbeddingError::model(format!("{dir}/config.json missing")))?;
        let get = |k: &str| config.get(k).and_then(Value::as_u64).map(|n| n as usize);
        let in_features = get("in_features").ok_or_else(|| EmbeddingError::model("Dense config without in_features"))?;
        let out_features = get("out_features").ok_or_else(|| EmbeddingError::model("Dense config without out_features"))?;
        let has_bias = config.get("bias").and_then(Value::as_bool).unwrap_or(true);
        let activation = Activation::parse(config.get("activation_function").and_then(Value::as_str).unwrap_or("Tanh"))?;

        let (weight, bias) = if let Some(path) = files.primary(&format!("{dir}/model.safetensors"))? {
            let w = safetensor(&path, "linear.weight")?.ok_or_else(|| EmbeddingError::model("linear.weight missing"))?;
            let b = if has_bias { safetensor(&path, "linear.bias")?.map(|b| b.1) } else { None };
            (w, b)
        } else if let Some(path) = files.primary(&format!("{dir}/pytorch_model.bin"))? {
            let mut tensors = torch_pickle::load_state_dict(&path)?;
            let w = tensors.remove("linear.weight").ok_or_else(|| EmbeddingError::model("linear.weight missing"))?;
            let b = if has_bias { tensors.remove("linear.bias").map(|t| t.data) } else { None };
            ((w.shape, w.data), b)
        } else {
            return Err(EmbeddingError::model(format!("{dir}: no model.safetensors or pytorch_model.bin")));
        };
        let (shape, weight) = weight;
        if shape != [out_features, in_features] {
            return Err(EmbeddingError::model(format!("Dense weight shape {shape:?} != [{out_features}, {in_features}]")));
        }
        if bias.as_ref().is_some_and(|b| b.len() != out_features) {
            return Err(EmbeddingError::model("Dense bias size mismatch"));
        }
        Ok(Self { in_features, out_features, weight, bias, activation })
    }

    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(EmbeddingError::model(format!("Dense expects {} features, got {}", self.in_features, x.len())));
        }
        Ok((0..self.out_features)
            .map(|o| {
                let w = &self.weight[o * self.in_features..(o + 1) * self.in_features];
                let mut y: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
                if let Some(b) = &self.bias {
                    y += b[o];
                }
                self.activation.apply(y)
            })
            .collect())
    }
}
