//! `ensemble` — o `EnsembleRetriever` do LangChain: vários retrievers (estratégias ou
//! providers injetados), fundidos por RRF com `weights`.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::hybrid;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    hybrid::fusion(ctx, query, true)
}
