//! `diversity` — resultados diversos: MMR sobre a busca semântica (como o `dense_mmr`), para
//! não devolver vários trechos quase iguais.

use chunking::Document;

use crate::context::Ctx;
use crate::deps::SearchType;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    ctx.dense(query, SearchType::Mmr)
}
