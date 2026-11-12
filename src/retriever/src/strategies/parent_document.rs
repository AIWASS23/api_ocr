//! `parent_document` — o `ParentDocumentRetriever`: busca trechos pequenos (precisos) e devolve
//! o documento PAI de cada um (mais contexto para o LLM), sem repetir pais. O pai vem de
//! `options.parent_store` (`{id: documento}`) pela chave `options.parent_id_key`
//! (`parent_id`); sem ele, de `metadata.parent_content` (o que a estratégia `parent_child` do
//! chunking grava com `include_parent_content`); senão, o próprio trecho.

use chunking::Document;
use serde_json::Value;

use crate::context::Ctx;
use crate::core::documents::{as_document, document_key, py_str};
use crate::error::Result;

/// Pai de um trecho, se houver.
fn parent_of(ctx: &Ctx, child: &Document, id_key: &str) -> Option<Document> {
    let pid = child.metadata.get(id_key).filter(|v| !v.is_null())?;
    if let Some(Value::Object(store)) = ctx.cfg.option("parent_store")
        && let Some(v) = store.get(&py_str(pid))
    {
        return Some(as_document(v.clone()));
    }
    let content = child.metadata.get("parent_content").and_then(Value::as_str)?;
    let mut metadata = child.metadata.clone();
    for k in ["parent_content", "child_index", "id", "retrieval_score", "retrieval_rank", "retrieval_strategy"] {
        metadata.remove(k);
    }
    metadata.insert("id".into(), pid.clone());
    Some(Document::with_metadata(content, metadata))
}

/// `_hierarchical`: pais únicos dos trechos da estratégia base; em `small_to_big`, mais os
/// vizinhos de cada trecho (`with_neighbor_loader`).
pub fn hierarchical(ctx: &Ctx, query: &str, mode: &str) -> Result<Vec<Document>> {
    let children = ctx.base_results(query)?;
    let id_key = ctx.cfg.option_str("parent_id_key").unwrap_or("parent_id").to_string();
    let mut out: Vec<Document> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for child in &children {
        let parent = parent_of(ctx, child, &id_key).unwrap_or_else(|| child.clone());
        if seen.insert(document_key(&parent)) {
            out.push(parent);
        }
        if mode == "small_to_big"
            && let Some(loader) = &ctx.env.deps.neighbor_loader
        {
            out.extend(loader(child));
        }
        if out.len() >= ctx.cfg.k {
            break;
        }
    }
    out.truncate(ctx.cfg.k);
    Ok(out)
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    hierarchical(ctx, query, "parent")
}
