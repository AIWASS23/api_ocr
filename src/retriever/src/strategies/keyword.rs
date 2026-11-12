//! `keyword` — busca por palavra-chave (full text): nota = quantas vezes os termos da consulta
//! aparecem no texto (em minúsculas); só documentos com nota positiva.

use chunking::Document;

use crate::context::Ctx;
use crate::core::text::tokens;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let terms = tokens(query);
    let docs = ctx.corpus()?;
    let scores: Vec<f64> = docs
        .iter()
        .map(|d| {
            let text = d.page_content.to_lowercase();
            terms.iter().map(|t| text.matches(t.as_str()).count()).sum::<usize>() as f64
        })
        .collect();
    Ok(ctx.rank(&docs, &scores, true, None))
}
