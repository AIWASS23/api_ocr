//! `adaptive` — escolhe a estratégia pela consulta. Com um roteador injetado
//! (`with_router`), ele decide entre `options.strategies`. Senão: consulta com números ou
//! códigos (dígitos, `-`, `_`, `/`, `.`) vai para `hybrid` (o lexical acha códigos exatos); as
//! demais para `dense`; sem embedder nem provider, `bm25`.

use chunking::Document;
use regex::Regex;

use crate::context::Ctx;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let selected = if let Some(router) = &ctx.env.deps.router {
        router(query, &ctx.cfg.option_strings("strategies"))
    } else if ctx.has_dense_source() {
        let codes = Regex::new(r"[\d\-_/.]").expect("regex válida");
        if codes.is_match(query) { "hybrid".into() } else { "dense".into() }
    } else {
        "bm25".into()
    };
    ctx.execute(query, &selected)
}
