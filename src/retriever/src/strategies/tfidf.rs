//! `tfidf` — TF-IDF do scikit-learn (palavras e pares de palavras, IDF suavizado, norma L2)
//! ajustado no acervo; nota = cosseno entre consulta e documento.

use chunking::Document;
use embedding::core::options::Options;
use embedding::core::vectorizer::{CountVectorizer, TfidfWeights};
use serde_json::{Map, json};

use crate::context::Ctx;
use crate::error::Result;

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    let docs = ctx.corpus()?;
    if docs.is_empty() {
        return Ok(Vec::new());
    }
    let opts = Map::from_iter([("ngram_range".to_string(), json!([1, 2]))]);
    let mut o = Options::new("tfidf", &opts);
    let vectorizer = CountVectorizer::from_options(&mut o, None)?;
    let mut weights = TfidfWeights::from_options(&mut o)?;
    let texts: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
    let (vocab, counts) = vectorizer.fit_transform(&texts)?;
    weights.fit(&counts, vocab.len());
    let rows = weights.transform(counts);
    let q = weights.transform(vectorizer.transform(&vocab, &[query])).remove(0);
    let mut qv = vec![0f64; vocab.len()];
    for (i, v) in q {
        qv[i] = v;
    }
    let scores: Vec<f64> = rows.iter().map(|r| r.iter().map(|&(i, v)| v * qv[i]).sum()).collect();
    Ok(ctx.rank(&docs, &scores, false, None))
}
