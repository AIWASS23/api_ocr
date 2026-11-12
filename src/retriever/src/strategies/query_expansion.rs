//! `query_expansion` — expansão de consulta: o LLM gera consultas com sinônimos e termos
//! relacionados (modo `expansion`), buscadas e fundidas como no `multi_query`.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::multi_query;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    multi_query::multi(ctx, query, "expansion")
}
