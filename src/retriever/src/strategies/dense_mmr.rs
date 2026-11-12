//! `dense_mmr` — busca semântica com Maximal Marginal Relevance: escolhe, um a um, o documento
//! mais relevante que menos se parece com os já escolhidos. `options.lambda_mult` (0,5) pesa
//! relevância × diversidade; no provider, `fetch_k` candidatos.

use chunking::Document;

use crate::context::Ctx;
use crate::deps::SearchType;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    ctx.dense(query, SearchType::Mmr)
}
