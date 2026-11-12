//! `knowledge_base` — base de conhecimento gerenciada (Bedrock Knowledge Bases, Vertex AI
//! Search, Azure AI Search, Kendra no Python), que já faz indexação e busca. A fonte é o
//! provider injetado.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::graph::provider_results;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    provider_results(ctx, query)
}
