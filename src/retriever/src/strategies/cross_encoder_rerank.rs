//! `cross_encoder_rerank` — reordena os candidatos com um cross-encoder (consulta e documento
//! juntos no mesmo modelo, que vê a interação entre os dois): o `rerank_model` em ONNX Runtime,
//! ou um reranker injetado.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::rerank::{self, Mode};

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let candidates = ctx.base_results(query)?;
    rerank::rerank(ctx, query, &candidates, Mode::CrossEncoder)
}
