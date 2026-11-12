//! `hyde` — Hypothetical Document Embeddings: o LLM escreve um documento curto que responderia
//! a pergunta, e é esse texto (mais parecido com os documentos que a pergunta) que é buscado.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::rephrase_query;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    rephrase_query::transformed(ctx, query, "hyde")
}
