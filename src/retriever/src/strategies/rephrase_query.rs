//! `rephrase_query` — o `RePhraseQueryRetriever`: o LLM reescreve a consulta para busca (sem
//! responder) e a reescrita é buscada com a estratégia base.

use chunking::Document;

use crate::context::{Component, Ctx};
use crate::error::Result;

/// Consulta transformada pelo `QueryGenerator` injetado ou pelo LLM, buscada com a base.
pub fn transformed(ctx: &Ctx, query: &str, mode: &str) -> Result<Vec<Document>> {
    let text = match &ctx.env.deps.query_generator {
        Some(g) => g.generate(query, mode, None)?.into_iter().next().unwrap_or_default(),
        None => {
            let prompt = match mode {
                "rephrase" => format!("Reescreva esta consulta para busca, sem responder:\n{query}"),
                "hyde" => format!("Escreva um documento curto que responderia esta pergunta:\n{query}"),
                _ => format!("Gere uma pergunta mais geral que ajude a responder:\n{query}"),
            };
            ctx.call_llm(&prompt)?
        }
    };
    ctx.component(chunking::core::pystr::strip(&text), &Component::Strategy(ctx.base_strategy().into()))
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    transformed(ctx, query, "rephrase")
}
