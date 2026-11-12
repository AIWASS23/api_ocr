//! As 44 estratégias de busca, uma por arquivo (a ordem segue o catálogo do Python).

use chunking::Document;

use crate::context::Ctx;
use crate::error::{RetrievalError, Result};

pub mod dense;
pub mod dense_mmr;
pub mod similarity_threshold;
pub mod multimodal;
pub mod sparse;
pub mod bm25;
pub mod bm25_plus;
pub mod tfidf;
pub mod keyword;
pub mod boolean;
pub mod fuzzy;
pub mod metadata;
pub mod hybrid;
pub mod weighted_hybrid;
pub mod rrf;
pub mod ensemble;
pub mod multi_query;
pub mod query_expansion;
pub mod rephrase_query;
pub mod hyde;
pub mod step_back;
pub mod decomposition;
pub mod self_query;
pub mod parent_document;
pub mod multi_vector;
pub mod small_to_big;
pub mod contextual_compression;
pub mod rerank;
pub mod cross_encoder_rerank;
pub mod colbert_rerank;
pub mod llm_rerank;
pub mod diversity;
pub mod time_weighted;
pub mod recency;
pub mod graph;
pub mod knowledge_graph;
pub mod sql;
pub mod web_search;
pub mod knowledge_base;
pub mod adaptive;
pub mod router;
pub mod fallback;
pub mod cascade;
pub mod custom;

/// Executa a estratégia `name` (já normalizada).
pub fn run(name: &str, ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    match name {
        "dense" => dense::run(ctx, query),
        "dense_mmr" => dense_mmr::run(ctx, query),
        "similarity_threshold" => similarity_threshold::run(ctx, query),
        "multimodal" => multimodal::run(ctx, query),
        "sparse" => sparse::run(ctx, query),
        "bm25" => bm25::run(ctx, query),
        "bm25_plus" => bm25_plus::run(ctx, query),
        "tfidf" => tfidf::run(ctx, query),
        "keyword" => keyword::run(ctx, query),
        "boolean" => boolean::run(ctx, query),
        "fuzzy" => fuzzy::run(ctx, query),
        "metadata" => metadata::run(ctx, query),
        "hybrid" => hybrid::run(ctx, query),
        "weighted_hybrid" => weighted_hybrid::run(ctx, query),
        "rrf" => rrf::run(ctx, query),
        "ensemble" => ensemble::run(ctx, query),
        "multi_query" => multi_query::run(ctx, query),
        "query_expansion" => query_expansion::run(ctx, query),
        "rephrase_query" => rephrase_query::run(ctx, query),
        "hyde" => hyde::run(ctx, query),
        "step_back" => step_back::run(ctx, query),
        "decomposition" => decomposition::run(ctx, query),
        "self_query" => self_query::run(ctx, query),
        "parent_document" => parent_document::run(ctx, query),
        "multi_vector" => multi_vector::run(ctx, query),
        "small_to_big" => small_to_big::run(ctx, query),
        "contextual_compression" => contextual_compression::run(ctx, query),
        "rerank" => rerank::run(ctx, query),
        "cross_encoder_rerank" => cross_encoder_rerank::run(ctx, query),
        "colbert_rerank" => colbert_rerank::run(ctx, query),
        "llm_rerank" => llm_rerank::run(ctx, query),
        "diversity" => diversity::run(ctx, query),
        "time_weighted" => time_weighted::run(ctx, query),
        "recency" => recency::run(ctx, query),
        "graph" => graph::run(ctx, query),
        "knowledge_graph" => knowledge_graph::run(ctx, query),
        "sql" => sql::run(ctx, query),
        "web_search" => web_search::run(ctx, query),
        "knowledge_base" => knowledge_base::run(ctx, query),
        "adaptive" => adaptive::run(ctx, query),
        "router" => router::run(ctx, query),
        "fallback" => fallback::run(ctx, query),
        "cascade" => cascade::run(ctx, query),
        "custom" => custom::run(ctx, query),
        other => Err(RetrievalError::UnsupportedStrategy(other.into(), crate::config::SUPPORTED_STRATEGIES.join(", "))),
    }
}
