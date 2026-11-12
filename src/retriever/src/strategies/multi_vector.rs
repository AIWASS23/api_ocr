//! `multi_vector` — o `MultiVectorRetriever`: o índice guarda várias representações por
//! documento (resumos, perguntas hipotéticas, trechos), todas apontando para o mesmo
//! `parent_id`; a busca acha a representação e devolve o documento original.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::parent_document;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    parent_document::hierarchical(ctx, query, "multi_vector")
}
