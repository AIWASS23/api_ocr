//! Embeddings por token (antes do pooling) para o *late chunking*: o documento inteiro passa
//! pelo modelo de uma vez e cada chunk recebe a média dos vetores dos seus tokens, então o
//! vetor de cada chunk "enxerga" o resto do documento.
//!
//! [`OnnxTokenEmbedder`] (feature `onnx`) roda um modelo ONNX de embeddings (`model.onnx` com
//! saída `last_hidden_state`) com ONNX Runtime, em processo. Textos maiores que o limite do
//! modelo são processados em janelas sobrepostas.

use crate::error::Result;

/// Vetores por token e as posições (em bytes) de cada token no texto original.
#[derive(Debug, Clone, Default)]
pub struct TokenEmbeddings {
    pub offsets: Vec<(usize, usize)>,
    pub vectors: Vec<Vec<f32>>,
}

pub trait TokenEmbedder: Send + Sync {
    fn embed_tokens(&self, text: &str) -> Result<TokenEmbeddings>;
    /// Nome do modelo, gravado no metadado `embedding_model`.
    fn model_name(&self) -> &str;
}

#[cfg(feature = "onnx")]
pub use onnx::OnnxTokenEmbedder;

#[cfg(feature = "onnx")]
mod onnx {
    use std::sync::Mutex;

    use ort::session::Session;
    use ort::value::Tensor;

    use super::{TokenEmbedder, TokenEmbeddings};
    use crate::error::{ChunkError, Result};
    use crate::models::tokenizer::hf_model_file;

    pub struct OnnxTokenEmbedder {
        session: Mutex<Session>,
        tokenizer: tokenizers::Tokenizer,
        name: String,
        /// Tokens por janela (inclui os especiais).
        max_tokens: usize,
        needs_token_type_ids: bool,
    }

    /// Tokens de uma janela: posição absoluta (bytes) e vetor.
    type WindowTokens = Vec<((usize, usize), Vec<f32>)>;

    fn model_err(e: impl std::fmt::Display) -> ChunkError {
        ChunkError::Model(e.to_string())
    }

    impl OnnxTokenEmbedder {
        /// `name_or_path`: repositório do Hub ou diretório com `tokenizer.json` e
        /// `onnx/model.onnx` (ou `model.onnx`).
        pub fn load(name_or_path: &str, max_tokens: usize) -> Result<Self> {
            let tokenizer_path = hf_model_file(name_or_path, "tokenizer.json")?;
            let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(model_err)?;
            tokenizer.with_truncation(None).map_err(model_err)?.with_padding(None);
            let model_path = hf_model_file(name_or_path, "onnx/model.onnx").or_else(|_| hf_model_file(name_or_path, "model.onnx"))?;
            let session = Session::builder().map_err(model_err)?.commit_from_file(&model_path).map_err(model_err)?;
            let needs_token_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");
            Ok(Self { session: Mutex::new(session), tokenizer, name: name_or_path.to_string(), max_tokens: max_tokens.max(8), needs_token_type_ids })
        }

        /// Uma janela: tokeniza `text[start..end]` com especiais e devolve (offsets absolutos, vetores).
        fn run_window(&self, text: &str, start: usize, end: usize) -> Result<WindowTokens> {
            let enc = self.tokenizer.encode(&text[start..end], true).map_err(model_err)?;
            let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
            let n = ids.len();
            let mask = vec![1i64; n];
            let types = vec![0i64; n];
            let mut session = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let ids_t = Tensor::from_array(([1usize, n], ids)).map_err(model_err)?;
            let mask_t = Tensor::from_array(([1usize, n], mask)).map_err(model_err)?;
            let outputs = if self.needs_token_type_ids {
                let types_t = Tensor::from_array(([1usize, n], types)).map_err(model_err)?;
                session.run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t, "token_type_ids" => types_t])
            } else {
                session.run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t])
            }
            .map_err(model_err)?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(model_err)?;
            let dim = *shape.last().ok_or_else(|| model_err("saída sem dimensões"))? as usize;
            let mut out = Vec::new();
            for (i, &(s, e)) in enc.get_offsets().iter().enumerate() {
                // Tokens especiais têm offset vazio e ficam de fora do pooling.
                if s == e || enc.get_special_tokens_mask()[i] == 1 {
                    continue;
                }
                out.push(((start + s, start + e), data[i * dim..(i + 1) * dim].to_vec()));
            }
            Ok(out)
        }
    }

    impl TokenEmbedder for OnnxTokenEmbedder {
        fn embed_tokens(&self, text: &str) -> Result<TokenEmbeddings> {
            // Janelas por token do texto inteiro (sem especiais), com 1/8 de sobreposição;
            // cada token fica com o vetor da janela em que está mais longe da borda.
            let full = self.tokenizer.encode(text, false).map_err(model_err)?;
            let offs = full.get_offsets().to_vec();
            if offs.is_empty() {
                return Ok(TokenEmbeddings::default());
            }
            let window = self.max_tokens.saturating_sub(2).max(4);
            let overlap = window / 8;
            let mut best: Vec<Option<(usize, Vec<f32>)>> = vec![None; offs.len()];
            let mut w_start = 0;
            loop {
                let w_end = (w_start + window).min(offs.len());
                let span = (offs[w_start].0, offs[w_end - 1].1);
                let tokens = self.run_window(text, span.0, span.1)?;
                let center = (w_start + w_end) / 2;
                for (k, slot) in best.iter_mut().enumerate().take(w_end).skip(w_start) {
                    let dist = k.abs_diff(center);
                    // Vetor do token k nesta janela: o token da janela com a mesma posição.
                    if let Some((_, v)) = tokens.iter().find(|((s, e), _)| *s < offs[k].1 && offs[k].0 < *e)
                        && slot.as_ref().is_none_or(|(d, _)| dist < *d)
                    {
                        *slot = Some((dist, v.clone()));
                    }
                }
                if w_end == offs.len() {
                    break;
                }
                w_start = w_end - overlap;
            }
            let mut result = TokenEmbeddings::default();
            for (k, slot) in best.into_iter().enumerate() {
                if let Some((_, v)) = slot {
                    result.offsets.push(offs[k]);
                    result.vectors.push(v);
                }
            }
            Ok(result)
        }

        fn model_name(&self) -> &str {
            &self.name
        }
    }
}
