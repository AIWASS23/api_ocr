//! `huggingface` — qualquer encoder do Hugging Face (BERT, RoBERTa, XLM-R...) sem a configuração
//! do sentence-transformers (`HuggingFacePoolingAdapter` do chatbot): tokeniza com corte em
//! `max_length`, roda o modelo e faz o pooling pedido sobre o `last_hidden_state`.
//!
//! Padrão: `sentence-transformers/all-MiniLM-L6-v2`. Opções: `pooling` (`mean`, `max` ou `cls`),
//! `max_length` (512), `model_options` (`revision`; os kwargs do `AutoModel` sem efeito em ONNX são
//! aceitos e ignorados), `onnx_file`.

use crate::config::EmbeddingConfig;
use crate::core::hub::ModelFiles;
use crate::core::options::Options;
use crate::core::st::TEXT_ONNX;
use crate::core::text_model::{TextInput, TextModel};
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pool {
    Mean,
    Max,
    Cls,
}

pub struct HuggingFaceEmbedder {
    name: String,
    model: TextModel,
    output: String,
    pooling: Pool,
    batch_size: usize,
}

pub fn load(config: &EmbeddingConfig) -> Result<HuggingFaceEmbedder> {
    let mut o = Options::new("huggingface", &config.options);
    let pooling = match o.string("pooling")?.as_deref() {
        None | Some("mean") => Pool::Mean,
        Some("max") => Pool::Max,
        Some("cls") => Pool::Cls,
        Some(_) => return Err(EmbeddingError::config("pooling must be 'mean', 'max' or 'cls'")),
    };
    let max_length = o.usize("max_length")?.unwrap_or(512);
    let onnx_file = o.string("onnx_file")?;
    let mut mo = Options::new("model_options", &o.object("model_options")?);
    let revision = mo.string("revision")?;
    mo.ignore(&["trust_remote_code", "token", "cache_dir", "local_files_only", "torch_dtype", "dtype", "tokenizer_options", "attn_implementation"]);
    mo.finish()?;
    o.finish()?;
    let name = config.model_name.clone().ok_or_else(|| EmbeddingError::config("model_name is required for Hugging Face embeddings"))?;
    let files = ModelFiles::open(&name, revision.as_deref())?;
    let model = TextModel::load(&files, TEXT_ONNX, onnx_file.as_deref(), &config.device, max_length)?;
    let output = model.onnx.pick_output(&["last_hidden_state"], 3)?;
    Ok(HuggingFaceEmbedder { name, model, output, pooling, batch_size: config.batch_size })
}

impl Embedder for HuggingFaceEmbedder {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn embed(&self, texts: &[&str], _input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size.max(1)) {
            let inputs: Vec<TextInput> = batch.iter().map(|t| TextInput::Single(t)).collect();
            let enc = self.model.encode(&inputs)?;
            let res = self.model.run(&enc, Vec::new(), &[self.output.as_str()])?;
            let [_, seq, hidden] = res[0].shape.as_slice() else {
                return Err(EmbeddingError::model(format!("unexpected last_hidden_state shape {:?}", res[0].shape)));
            };
            let (seq, hidden) = (*seq, *hidden);
            for r in 0..enc.rows {
                let tokens = &res[0].data[r * seq * hidden..(r + 1) * seq * hidden];
                let mask = &enc.mask[r * enc.seq..(r + 1) * enc.seq];
                let row = |t: usize| &tokens[t * hidden..(t + 1) * hidden];
                out.push(match self.pooling {
                    // tokens[:, 0], qualquer que seja o padding.
                    Pool::Cls => row(0).to_vec(),
                    Pool::Max => {
                        let mut v = vec![f32::NEG_INFINITY; hidden];
                        for t in (0..seq).filter(|&t| mask[t] != 0) {
                            v.iter_mut().zip(row(t)).for_each(|(a, &b)| *a = a.max(b));
                        }
                        v
                    }
                    Pool::Mean => {
                        let mut v = vec![0f32; hidden];
                        for t in (0..seq).filter(|&t| mask[t] != 0) {
                            v.iter_mut().zip(row(t)).for_each(|(a, &b)| *a += b);
                        }
                        // clamp(min=1) do Python.
                        let n = (mask.iter().filter(|&&m| m != 0).count() as f32).max(1.0);
                        v.into_iter().map(|x| x / n).collect()
                    }
                });
            }
        }
        Ok(out)
    }
}
