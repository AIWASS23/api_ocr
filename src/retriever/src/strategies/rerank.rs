//! `rerank` — busca `fetch_k` candidatos com a estratégia base e reordena com um modelo mais
//! preciso (reranker injetado ou o cross-encoder `rerank_model`, `BAAI/bge-reranker-base`),
//! devolvendo `top_n`.

use chunking::Document;
use regex::Regex;

use crate::context::Ctx;
use crate::error::{RetrievalError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Default,
    CrossEncoder,
    Colbert,
    Llm,
}

/// Nota de 0 a 10 dada pelo LLM (primeiro número da resposta).
fn llm_score(ctx: &Ctx, query: &str, doc: &Document) -> Result<f64> {
    let reply = ctx.call_llm(&format!(
        "Avalie a relevância de 0 a 10 e responda somente o número.\nConsulta: {query}\nDocumento: {}",
        doc.page_content
    ))?;
    let re = Regex::new(r"-?\d+(?:[.,]\d+)?").expect("regex válida");
    Ok(re.find(&reply).and_then(|m| m.as_str().replace(',', ".").parse().ok()).unwrap_or(0.0))
}

/// Reordena `documents` (`_rerank`).
pub fn rerank(ctx: &Ctx, query: &str, documents: &[Document], mode: Mode) -> Result<Vec<Document>> {
    let top_n = ctx.cfg.top_n;
    let reranker = match (&ctx.env.deps.reranker, mode) {
        (Some(r), _) => r.clone(),
        (None, Mode::Colbert) => {
            return Err(RetrievalError::missing("colbert_rerank requires an injected reranker (with_reranker) with a ColBERT model"));
        }
        (None, Mode::Llm) => {
            let scores: Vec<f64> = documents.iter().map(|d| llm_score(ctx, query, d)).collect::<Result<_>>()?;
            return Ok(ctx.rank(documents, &scores, false, Some(top_n)));
        }
        (None, _) => ctx.cross_encoder()?,
    };
    let mut out = reranker.rerank(query, documents, top_n)?;
    out.truncate(top_n);
    Ok(out)
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let candidates = ctx.base_results(query)?;
    rerank(ctx, query, &candidates, Mode::Default)
}
