//! `fuzzy` — busca aproximada: semelhança de Ratcliff/Obershelp (o `difflib.SequenceMatcher` do
//! Python) entre a consulta e o texto inteiro de cada documento, em minúsculas. Tolera erros de
//! digitação em acervos de textos curtos (títulos, nomes, FAQs).

use chunking::Document;

use crate::context::Ctx;
use crate::core::text::ratio;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let q = query.to_lowercase();
    let docs = ctx.corpus()?;
    let scores: Vec<f64> = docs.iter().map(|d| ratio(&q, &d.page_content.to_lowercase())).collect();
    Ok(ctx.rank(&docs, &scores, false, None))
}
