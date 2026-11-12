//! `router` — um roteador injetado (`with_router`: regras, classificador ou LLM) escolhe a
//! estratégia de cada consulta entre `options.strategies`.

use chunking::Document;

use crate::context::Ctx;
use crate::error::{RetrievalError, Result};

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let router = ctx
        .env
        .deps
        .router
        .as_ref()
        .ok_or_else(|| RetrievalError::config("router strategy requires a router (RetrievalManager::with_router)"))?;
    let selected = router(query, &ctx.cfg.option_strings("strategies"));
    ctx.execute(query, &selected)
}
