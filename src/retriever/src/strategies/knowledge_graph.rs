//! `knowledge_graph` — busca num grafo de conhecimento (triplas entidade–relação–entidade,
//! consultas Cypher/SPARQL geradas a partir da pergunta). A fonte é o provider injetado.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::graph::provider_results;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    provider_results(ctx, query)
}
