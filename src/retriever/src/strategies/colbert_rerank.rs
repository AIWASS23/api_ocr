//! `colbert_rerank` — reordena com ColBERT (interação tardia: máximo de similaridade por token).
//! Como no Python (RAGatouille), o modelo é externo: injete um `Reranker` com
//! `with_reranker`.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::rerank::{self, Mode};

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let candidates = ctx.base_results(query)?;
    rerank::rerank(ctx, query, &candidates, Mode::Colbert)
}
