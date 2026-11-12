//! `cascade` — pipeline em estágios (`options.stages`, padrão `["dense", "rerank"]`). O primeiro
//! busca; os seguintes refinam o resultado anterior: rerankers reordenam,
//! `contextual_compression` comprime, qualquer outra estratégia é executada e fundida (RRF) com
//! o que já havia.

use chunking::Document;

use crate::config::normalize_strategy;
use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::contextual_compression::compress;
use crate::strategies::rerank::{Mode, rerank};

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let mut stages = ctx.cfg.option_strings("stages");
    if stages.is_empty() {
        stages = vec!["dense".into(), "rerank".into()];
    }
    let mut result: Vec<Document> = Vec::new();
    for (index, stage) in stages.iter().enumerate() {
        let selected = normalize_strategy(stage)?;
        result = if index == 0 {
            ctx.execute(query, selected)?
        } else {
            match selected {
                "rerank" => rerank(ctx, query, &result, Mode::Default)?,
                "cross_encoder_rerank" => rerank(ctx, query, &result, Mode::CrossEncoder)?,
                "colbert_rerank" => rerank(ctx, query, &result, Mode::Colbert)?,
                "llm_rerank" => rerank(ctx, query, &result, Mode::Llm)?,
                "contextual_compression" => compress(ctx, query, &result)?,
                other => {
                    let next = ctx.execute(query, other)?;
                    ctx.fuse(&[result, next])
                }
            }
        };
    }
    result.truncate(ctx.cfg.k);
    Ok(result)
}
