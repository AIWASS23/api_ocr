//! `hybrid` — busca híbrida: roda os componentes (`options.components` ou `retrievers`;
//! padrão `dense` + `bm25`, mais providers injetados) e funde as listas por Reciprocal Rank
//! Fusion, `Σ 1/(rrf_k + posição)` (`options.rrf_k`, 60), sem pesos. Documentos repetidos entre
//! as listas viram um só (chave: `id`, `source`, `url`... ou o texto).

use chunking::Document;

use crate::context::Ctx;
use crate::core::rank::reciprocal_rank_fusion;
use crate::error::{RetrievalError, Result};

/// Fusão dos componentes; `weighted` usa `weights` (um por componente) quando há.
pub fn fusion(ctx: &Ctx, query: &str, weighted: bool) -> Result<Vec<Document>> {
    let components = ctx.components();
    let rankings: Vec<Vec<Document>> = components.iter().map(|c| ctx.component(query, c)).collect::<Result<_>>()?;
    let weights = if weighted && !ctx.cfg.weights.is_empty() { ctx.cfg.weights.clone() } else { vec![1.0; rankings.len()] };
    if weights.len() != rankings.len() {
        return Err(RetrievalError::config("weights must contain one value per retrieval component"));
    }
    Ok(reciprocal_rank_fusion(&rankings, &weights, ctx.cfg.option_f64("rrf_k", 60.0), ctx.cfg.k))
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    fusion(ctx, query, false)
}
