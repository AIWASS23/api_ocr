//! `fallback` — tenta as estratégias de `options.strategies` (`["dense", "bm25"]`) em ordem e
//! devolve o primeiro resultado não vazio; falhas passam para a próxima. Se nenhuma devolver
//! algo, erro com os motivos.

use chunking::Document;

use crate::context::Ctx;
use crate::error::{RetrievalError, Result};

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let mut strategies = ctx.cfg.option_strings("strategies");
    if strategies.is_empty() {
        strategies = vec!["dense".into(), "bm25".into()];
    }
    let mut failures = Vec::new();
    for strategy in &strategies {
        match ctx.execute(query, strategy) {
            Ok(mut result) if !result.is_empty() => {
                result.truncate(ctx.cfg.k);
                return Ok(result);
            }
            Ok(_) => {}
            Err(e) => failures.push(e.to_string()),
        }
    }
    Err(RetrievalError::AllFailed(failures.join("; ")))
}
