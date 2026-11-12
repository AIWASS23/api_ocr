//! `metadata` — só filtro por metadado (`filters`): os primeiros `k` documentos que passam, sem
//! olhar o texto da consulta.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;

pub fn run(ctx: &Ctx, _query: &str) -> Result<Vec<Document>> {
    Ok(ctx.corpus()?.into_iter().take(ctx.cfg.k).collect())
}
