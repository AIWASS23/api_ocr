//! `sql` — dados estruturados: a pergunta vira SQL (text-to-SQL) e as linhas viram documentos.
//! A fonte é o provider injetado, que conhece o banco e o esquema.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::graph::provider_results;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    provider_results(ctx, query)
}
