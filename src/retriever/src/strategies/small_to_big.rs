//! `small_to_big` — busca em trechos pequenos e devolve o pai (como o `parent_document`) mais
//! os trechos vizinhos de cada achado, quando há um `with_neighbor_loader`.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::parent_document;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    parent_document::hierarchical(ctx, query, "small_to_big")
}
