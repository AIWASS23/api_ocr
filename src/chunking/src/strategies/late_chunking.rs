//! `late_chunking` (nova) — *late chunking* (Jina AI): o documento inteiro passa pelo modelo de
//! embeddings de uma vez e só depois é dividido. O vetor de cada chunk é a média dos vetores dos
//! seus tokens, calculados com o documento todo como contexto; um trecho como "ele foi
//! reeleito" fica perto da pessoa citada três parágrafos antes.
//!
//! Os limites dos chunks vêm de uma estratégia base. Cada chunk sai com o vetor em
//! `metadata.embedding`, pronto para indexar no lugar do embedding comum. Precisa de um modelo
//! de embeddings por token (`OnnxTokenEmbedder`, feature `onnx`, ou injeção).
//!
//! Opções: `base_strategy` (`"sentence"`), `normalize` (`true`: vetor com norma 1).
//! Metadados: `embedding`, `embedding_model`, `start_index`, `end_index` (posições em
//! caracteres).

use std::sync::Arc;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::Document;
use crate::error::{ChunkError, Result};
use crate::models::token_embeddings::TokenEmbedder;

pub struct ChunkLateChunking {
    sizing: Sizing,
    base: Box<dyn Chunker>,
    embedder: Arc<dyn TokenEmbedder>,
    normalize: bool,
}

/// Posição (em bytes) de cada chunk no texto, em ordem. Chunks normalizados são trechos do
/// original; se um não for (ex.: separador reinserido), tenta pelo começo e pelo fim.
pub(crate) fn locate(text: &str, chunks: &[String]) -> Vec<Option<(usize, usize)>> {
    let mut pos = 0;
    chunks
        .iter()
        .map(|c| {
            if let Some(i) = text[pos..].find(c.as_str()) {
                let s = pos + i;
                pos = s + c.len().min(1);
                return Some((s, s + c.len()));
            }
            let head: String = c.chars().take(24).collect();
            let tail: String = { let v: Vec<char> = c.chars().collect(); v[v.len().saturating_sub(24)..].iter().collect() };
            let s = text[pos..].find(&head).map(|i| pos + i)?;
            let e = text[s..].find(&tail).map(|i| s + i + tail.len())?;
            pos = s + 1;
            Some((s, e))
        })
        .collect()
}

impl ChunkLateChunking {
    pub fn new(sizing: Sizing, base: Box<dyn Chunker>, embedder: Arc<dyn TokenEmbedder>, mut o: Options) -> Result<Self> {
        let normalize = o.bool("normalize", true)?;
        o.finish()?;
        Ok(Self { sizing, base, embedder, normalize })
    }
}

impl Chunker for ChunkLateChunking {
    crate::sizing_accessors!("late_chunking");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        self.base.chunk(text)
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let chunks = self.base.chunk_documents(text)?;
        let texts: Vec<String> = chunks.iter().map(|d| d.page_content.clone()).collect();
        let spans = locate(text, &texts);
        let tokens = self.embedder.embed_tokens(text)?;
        let mut out = Vec::new();
        for (doc, span) in chunks.into_iter().zip(spans) {
            let (s, e) = span.ok_or_else(|| ChunkError::Model("late_chunking: chunk não encontrado no texto original".into()))?;
            let vecs: Vec<&Vec<f32>> = tokens
                .offsets
                .iter()
                .zip(&tokens.vectors)
                .filter(|((ts, te), _)| *ts < e && s < *te)
                .map(|(_, v)| v)
                .collect();
            let Some(first) = vecs.first() else {
                return Err(ChunkError::Model("late_chunking: nenhum token do modelo cobre o chunk".into()));
            };
            let mut pooled = vec![0f32; first.len()];
            for v in &vecs {
                for (p, x) in pooled.iter_mut().zip(v.iter()) {
                    *p += *x;
                }
            }
            let n = vecs.len() as f32;
            pooled.iter_mut().for_each(|p| *p /= n);
            if self.normalize {
                let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    pooled.iter_mut().for_each(|p| *p /= norm);
                }
            }
            let mut meta = doc.metadata.clone();
            meta.insert("embedding".into(), serde_json::to_value(&pooled).map_err(|e| ChunkError::Json(e.to_string()))?);
            meta.insert("embedding_model".into(), self.embedder.model_name().into());
            meta.insert("start_index".into(), (pystr::len(&text[..s]) as u64).into());
            meta.insert("end_index".into(), (pystr::len(&text[..e]) as u64).into());
            out.push(Document::with_metadata(doc.page_content, meta));
        }
        Ok(out)
    }
}
