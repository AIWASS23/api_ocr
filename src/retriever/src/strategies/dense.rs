//! `dense` — busca semântica: cosseno entre o embedding da consulta e o de cada documento
//! (acervo local + embedder) ou busca de similaridade no provider. `score_threshold` descarta
//! notas abaixo do limiar.

use chunking::Document;

use crate::context::Ctx;
use crate::deps::SearchType;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    ctx.dense(query, SearchType::Similarity)
}
