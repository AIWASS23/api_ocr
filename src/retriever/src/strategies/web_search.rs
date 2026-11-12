//! `web_search` — resultados de uma busca na web (Tavily, Wikipedia, arXiv... no Python). A
//! fonte é o provider injetado; este crate não faz requisições à web por conta própria.

use chunking::Document;

use crate::context::Ctx;
use crate::error::Result;
use crate::strategies::graph::provider_results;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    provider_results(ctx, query)
}
