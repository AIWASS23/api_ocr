//! `weighted_hybrid` — busca híbrida com pesos: RRF em que cada componente conta `weights[i]`
//! (ex.: `[0.7, 0.3]` para `dense` e `bm25`).

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::hybrid;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    hybrid::fusion(ctx, query, true)
}
