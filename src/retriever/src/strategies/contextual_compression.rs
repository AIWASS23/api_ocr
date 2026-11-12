//! `contextual_compression` — o `ContextualCompressionRetriever`: busca `fetch_k` candidatos e
//! reduz cada um ao trecho relevante para a consulta (compressor injetado ou LLM), descartando
//! os que ficam vazios; devolve `top_n`. Menos texto irrelevante no prompt.

use chunking::Document;
use serde_json::Value;

use crate::context::Ctx;
use crate::error::Result;

/// Compressão de uma lista (`_compress`).
pub fn compress(ctx: &Ctx, query: &str, documents: &[Document]) -> Result<Vec<Document>> {
    if let Some(c) = &ctx.env.deps.compressor {
        let mut out = c.compress(query, documents)?;
        out.truncate(ctx.cfg.top_n);
        return Ok(out);
    }
    let mut out = Vec::new();
    for d in documents {
        let prompt = format!("Extraia somente o trecho relevante para a consulta. Consulta: {query}\nDocumento: {}", d.page_content);
        let text = chunking::core::pystr::strip(&ctx.call_llm(&prompt)?).to_string();
        if !text.is_empty() {
            let mut metadata = d.metadata.clone();
            metadata.insert("compressed".into(), Value::Bool(true));
            out.push(Document::with_metadata(text, metadata));
        }
    }
    out.truncate(ctx.cfg.top_n);
    Ok(out)
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let candidates = ctx.base_results(query)?;
    compress(ctx, query, &candidates)
}
