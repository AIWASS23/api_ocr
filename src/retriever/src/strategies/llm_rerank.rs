//! `llm_rerank` — reordena com o LLM dando uma nota de 0 a 10 a cada candidato (o RankLLM do
//! Python), ou com um reranker injetado.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::rerank::{self, Mode};

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let candidates = ctx.base_results(query)?;
    rerank::rerank(ctx, query, &candidates, Mode::Llm)
}
