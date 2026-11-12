//! Reranking com cross-encoder (`HuggingFaceCrossEncoder` do Python) em ONNX Runtime: cada par
//! (consulta, documento) passa pelo modelo e o logit vira a nota (sigmoide quando há um só
//! rótulo, como o `CrossEncoder.predict`; a coluna 1 quando há dois). Modelo: `rerank_model`
//! (diretório local ou Hub, com o ONNX do repositório ou de um espelho).

use chunking::Document;
use embedding::core::hub::ModelFiles;
use embedding::core::text_model::{TextInput, TextModel};

use crate::core::rank::rank_scores;
use crate::deps::Reranker;
use crate::error::{RetrievalError, Result};

pub struct CrossEncoder {
    model: TextModel,
    output: String,
    batch_size: usize,
}

impl CrossEncoder {
    pub fn load(name: &str, device: &str) -> Result<Self> {
        let files = ModelFiles::open(name, None)?;
        let model = TextModel::load(&files, &["onnx/model.onnx", "model.onnx"], None, device, 512)?;
        let output = model.onnx.pick_output(&["logits"], 2)?;
        Ok(Self { model, output, batch_size: 32 })
    }

    /// Nota de cada par.
    pub fn score(&self, query: &str, documents: &[Document]) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(documents.len());
        for batch in documents.chunks(self.batch_size) {
            let inputs: Vec<TextInput> = batch.iter().map(|d| TextInput::Pair(query, &d.page_content)).collect();
            let enc = self.model.encode(&inputs)?;
            let res = self.model.run(&enc, Vec::new(), &[self.output.as_str()])?;
            let cols = *res[0].shape.last().ok_or_else(|| RetrievalError::model("cross-encoder without output"))?;
            for row in res[0].data.chunks(cols.max(1)) {
                out.push(match cols {
                    1 => 1.0 / (1.0 + (-row[0] as f64).exp()),
                    _ => row[1] as f64,
                });
            }
        }
        Ok(out)
    }
}

impl Reranker for CrossEncoder {
    fn rerank(&self, query: &str, documents: &[Document], top_n: usize) -> Result<Vec<Document>> {
        let scores = self.score(query, documents)?;
        Ok(rank_scores(documents, &scores, false, top_n))
    }
}
