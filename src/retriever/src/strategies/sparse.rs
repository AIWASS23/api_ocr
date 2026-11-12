//! `sparse` — busca esparsa: BM25 no acervo local (documentos ou provider `local`); sem acervo,
//! a busca do provider (bancos com vetor esparso, como no Python).

use chunking::Document;

use crate::context::Ctx;
use crate::deps::SearchType;
use crate::error::Result;
use crate::strategies::bm25;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    if !ctx.cfg.documents.is_empty() || ctx.cfg.provider == "local" {
        return bm25::score(ctx, query, false);
    }
    ctx.dense(query, SearchType::Similarity)
}
