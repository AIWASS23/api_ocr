//! `multi_query` — o `MultiQueryRetriever`: gera variações da consulta (LLM ou
//! `QueryGenerator` injetado; `options.query_count`, 3), busca cada uma com a estratégia base e
//! funde tudo por RRF. Pega documentos que a formulação original perderia.

use chunking::Document;
use chunking::core::pystr;

use crate::context::{Component, Ctx};
use crate::error::Result;

/// Remove marcadores de lista (`- `, `* `, `1. `, `2) `) do início de uma linha.
fn strip_marker(line: &str) -> String {
    let s = pystr::lstrip(line);
    let rest = if let Some(r) = s.strip_prefix(['-', '*']) {
        r
    } else {
        let digits = s.chars().take_while(char::is_ascii_digit).count();
        match s[digits..].chars().next() {
            Some('.' | ')') if digits > 0 => &s[digits + 1..],
            _ => s,
        }
    };
    pystr::strip(rest).to_string()
}

/// Consultas geradas (sem repetidas nem a original), no máximo `query_count`.
pub fn generate(ctx: &Ctx, query: &str, mode: &str) -> Result<Vec<String>> {
    let count = ctx.cfg.option("query_count").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
    let lines: Vec<String> = match &ctx.env.deps.query_generator {
        Some(g) => g.generate(query, mode, Some(count))?,
        None => {
            let prompt = format!("Gere {count} consultas de busca em linhas separadas. Modo: {mode}. Consulta original: {query}");
            pystr::splitlines(&ctx.call_llm(&prompt)?, false).into_iter().map(str::to_string).collect()
        }
    };
    let mut out: Vec<String> = Vec::new();
    for line in lines {
        let text = strip_marker(&line);
        if !text.is_empty() && !out.contains(&text) && text != query {
            out.push(text);
        }
    }
    out.truncate(count);
    Ok(out)
}

/// Busca a original e as geradas com a estratégia base e funde por RRF.
pub fn multi(ctx: &Ctx, query: &str, mode: &str) -> Result<Vec<Document>> {
    let mut queries = vec![query.to_string()];
    queries.extend(generate(ctx, query, mode)?);
    let base = Component::Strategy(ctx.base_strategy().into());
    let rankings: Vec<Vec<Document>> = queries.iter().map(|q| ctx.component(q, &base)).collect::<Result<_>>()?;
    Ok(ctx.fuse(&rankings))
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    multi(ctx, query, "variants")
}

#[cfg(test)]
mod tests {
    use super::strip_marker;

    #[test]
    fn list_markers() {
        assert_eq!(strip_marker("  - praias"), "praias");
        assert_eq!(strip_marker("2) dunas "), "dunas");
        assert_eq!(strip_marker("10. litoral"), "litoral");
        assert_eq!(strip_marker("2024 em Fortaleza"), "2024 em Fortaleza");
    }
}
