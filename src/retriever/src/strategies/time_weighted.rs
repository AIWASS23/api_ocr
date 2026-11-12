//! `time_weighted` — o `TimeWeightedVectorStoreRetriever`: candidatos da estratégia base com
//! nota = relevância + recência (`exp(−decay·idade em dias)`), favorecendo o que é recente.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::recency;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    recency::recency(ctx, query, true)
}
