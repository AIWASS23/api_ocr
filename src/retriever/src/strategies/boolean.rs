//! `boolean` — busca booleana: documentos com TODOS os termos da consulta (`options.operator =
//! "and"`, padrão) ou com ALGUM (`"or"`), na ordem do acervo, sem nota.

use std::collections::HashSet;

use chunking::Document;

use crate::context::Ctx;
use crate::core::text::tokens;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let terms: HashSet<String> = tokens(query).into_iter().collect();
    let all = ctx.cfg.option_str("operator").unwrap_or("and").to_lowercase() == "and";
    Ok(ctx
        .corpus()?
        .into_iter()
        .filter(|d| {
            let doc: HashSet<String> = tokens(&d.page_content).into_iter().collect();
            if all { terms.is_subset(&doc) } else { !terms.is_disjoint(&doc) }
        })
        .take(ctx.cfg.k)
        .collect())
}
