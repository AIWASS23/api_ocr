//! `graph` — busca num grafo (GraphRAG, travessia de vizinhos de entidades). Como no Python, a
//! fonte é o provider: injete um `SearchProvider` com o seu grafo (Neo4j, `graph_rag`...).

use chunking::Document;

use crate::context::Ctx;
use crate::deps::{SearchRequest, SearchType};
use crate::error::Result;

/// `_provider_results`: os `k` primeiros do provider, com os filtros (sem limiar de nota).
pub fn provider_results(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let provider = ctx.provider()?;
    let request = SearchRequest {
        query,
        vector: None,
        k: ctx.cfg.k,
        fetch_k: None,
        search_type: SearchType::Similarity,
        score_threshold: None,
        filters: &ctx.cfg.filters,
        collection: ctx.cfg.collection_name.as_deref(),
    };
    let mut docs = provider.search(&request)?;
    docs.truncate(ctx.cfg.k);
    Ok(docs)
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    provider_results(ctx, query)
}
