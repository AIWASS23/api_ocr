//! `decomposition` — decomposição: o LLM quebra uma pergunta composta em subperguntas (modo
//! `decomposition`); cada uma é buscada e as listas são fundidas por RRF.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::multi_query;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    multi_query::multi(ctx, query, "decomposition")
}
