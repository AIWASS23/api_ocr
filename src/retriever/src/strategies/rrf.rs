//! `rrf` — Reciprocal Rank Fusion dos componentes, com `weights` quando houver (o mesmo cálculo
//! do `weighted_hybrid`; o nome que a literatura usa).

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::hybrid;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    hybrid::fusion(ctx, query, true)
}
