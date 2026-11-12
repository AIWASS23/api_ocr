//! `step_back` — step-back prompting: o LLM gera uma pergunta mais geral, que traz o contexto
//! de fundo necessário para responder a específica.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::rephrase_query;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    rephrase_query::transformed(ctx, query, "step_back")
}
