//! `bm25_plus` — BM25+: o BM25 com um piso `delta` (1) por termo encontrado, para que documentos
//! longos não sejam penalizados demais.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::bm25;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    bm25::score(ctx, query, true)
}
