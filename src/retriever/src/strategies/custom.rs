//! `custom` — o seu próprio retriever, injetado com `RetrievalManager::with_custom` (no Python,
//! `options.retriever`/`executor`). Recebe a consulta, `k` e os filtros.

use chunking::Document;

use crate::context::Ctx;
use crate::deps::{SearchRequest, SearchType};
use crate::error::{RetrievalError, Result};

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let retriever = ctx
        .env
        .deps
        .custom
        .as_ref()
        .ok_or_else(|| RetrievalError::config("custom strategy requires a retriever (RetrievalManager::with_custom)"))?;
    let request = SearchRequest {
        query,
        vector: None,
        k: ctx.cfg.k,
        fetch_k: None,
        search_type: SearchType::Similarity,
        score_threshold: None,
        filters: &ctx.cfg.filters,
        collection: ctx.cfg.collection_name.as_deref(),
    };
    let mut docs = retriever.search(&request)?;
    docs.truncate(ctx.cfg.k);
    Ok(docs)
}
