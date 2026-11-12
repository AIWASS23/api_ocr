//! `similarity_threshold` — busca semântica que só devolve documentos com nota de pelo menos
//! `score_threshold` (no provider, `search_type = similarity_score_threshold`).

use chunking::Document;

use crate::context::Ctx;
use crate::deps::SearchType;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    ctx.dense(query, SearchType::SimilarityScoreThreshold)
}
