//! `recency` — os documentos mais recentes do acervo: nota `exp(−decay·idade em dias)` pela data
//! em `metadata[options.time_key]` (`timestamp`; número Unix ou ISO 8601), com
//! `options.decay_rate` (0,01). Sem data, nota 0.

use chunking::Document;

use crate::context::Ctx;
use crate::core::documents::score_of;
use crate::core::time::{now, timestamp};
use crate::error::Result;

/// `_recency`; com `weighted`, sobre os candidatos da estratégia base e somando a relevância.
pub fn recency(ctx: &Ctx, query: &str, weighted: bool) -> Result<Vec<Document>> {
    let docs = if weighted { ctx.base_results(query)? } else { ctx.corpus()? };
    let key = ctx.cfg.option_str("time_key").unwrap_or("timestamp").to_string();
    let decay = ctx.cfg.option_f64("decay_rate", 0.01);
    let now = now();
    let scores: Vec<f64> = docs
        .iter()
        .map(|d| {
            // Como no Python, data 0 conta como ausente.
            let recency = match timestamp(d.metadata.get(&key)).filter(|&t| t != 0.0) {
                Some(t) => (-decay * ((now - t) / 86_400.0).max(0.0)).exp(),
                None => 0.0,
            };
            if weighted { recency + score_of(d) } else { recency }
        })
        .collect();
    Ok(ctx.rank(&docs, &scores, false, None))
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    recency(ctx, query, false)
}
