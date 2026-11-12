//! Dependências falsas e determinísticas, iguais às de `fixtures/gen_parity.py`.

#![allow(dead_code)]

use std::sync::Arc;

use embedding::Embedder;
use retriever::core::documents::{as_document, py_str};
use retriever::{Document, RetrievalError, RetrievalManager, Result, SearchRequest};
use serde_json::{Map, Value, json};

pub const VOCABULARY: &[&str] =
    &["python", "rust", "banana", "busca", "híbrida", "densa", "lexical", "fruta", "sintaxe", "linguagem", "tipagem", "praia"];

pub fn fake_embedder() -> Arc<dyn Embedder> {
    embedding::from_fn("fake-counts", |texts, _| {
        Ok(texts
            .iter()
            .map(|t| {
                let lower = t.to_lowercase();
                VOCABULARY.iter().map(|w| lower.matches(w).count() as f32).collect()
            })
            .collect())
    })
}

fn after_document(prompt: &str) -> &str {
    prompt.split_once("Documento: ").map(|(_, d)| d).unwrap_or("")
}

pub fn fake_llm(prompt: &str) -> Result<String> {
    if prompt.starts_with("Avalie a relevância") {
        let doc = after_document(prompt).to_lowercase();
        let hits: usize = ["python", "busca", "rust"].iter().map(|w| doc.matches(w).count()).sum();
        return Ok(format!("Nota: {},5 de 10", hits * 2));
    }
    if prompt.starts_with("Extraia somente") {
        let doc = after_document(prompt);
        if doc.to_lowercase().contains("banana") {
            return Ok("  ".into());
        }
        return Ok(format!("  {}  ", doc.split_whitespace().take(3).collect::<Vec<_>>().join(" ")));
    }
    if prompt.contains("Modo: ") {
        return Ok("1. python sintaxe\n- busca densa\n* python\n\n2) busca densa\nfruta".into());
    }
    if prompt.starts_with("Reescreva") {
        return Ok("  rust sintaxe  ".into());
    }
    if prompt.starts_with("Escreva um documento") {
        return Ok("Busca híbrida usa busca densa e lexical.".into());
    }
    if prompt.starts_with("Gere uma pergunta mais geral") {
        return Ok("linguagem de programação".into());
    }
    Err(RetrievalError::model(format!("prompt inesperado: {prompt:?}")))
}

pub fn query_generator(query: &str, mode: &str, count: Option<usize>) -> Result<Vec<String>> {
    Ok(match count {
        None => vec![match mode {
            "rephrase" => "rust tipagem",
            "hyde" => "fruta amarela",
            _ => "busca",
        }
        .to_string()],
        Some(n) => {
            let mut v = vec![format!("{query} {mode}"), "busca lexical".into(), query.to_string(), "fruta".into()];
            v.truncate(n);
            v
        }
    })
}

pub fn reranker(_: &str, documents: &[Document], top_n: usize) -> Result<Vec<Document>> {
    Ok(documents.iter().rev().take(top_n).cloned().collect())
}

pub fn compressor(_: &str, documents: &[Document]) -> Result<Vec<Document>> {
    Ok(documents.iter().filter(|d| d.page_content.contains('a')).take(1).cloned().collect())
}

pub fn query_constructor(query: &str) -> Result<(String, Map<String, Value>)> {
    let filters = Map::from_iter([("kind".to_string(), json!("code"))]);
    Ok((query.replace("código", "").trim().to_string(), filters))
}

pub fn neighbor_loader(document: &Document) -> Vec<Document> {
    let id = document.metadata.get("id").map(py_str).unwrap_or_default();
    vec![Document::new(format!("Vizinho de {id}")).insert("id", format!("n-{id}"))]
}

pub fn router(query: &str, strategies: &[String]) -> String {
    let pick = if query.contains("fruta") { strategies.last() } else { strategies.first() };
    pick.cloned().unwrap_or_default()
}

/// Provider/custom: devolve o acervo em ordem inversa.
pub fn reversed_corpus(documents: &[Value]) -> impl Fn(&SearchRequest) -> Result<Vec<Document>> + Send + Sync + 'static {
    let docs: Vec<Document> = documents.iter().rev().cloned().map(as_document).collect();
    move |_: &SearchRequest| Ok(docs.clone())
}

/// Aplica as dependências pedidas por nome (as mesmas de `DEPS` no gerador).
pub fn inject(mut m: RetrievalManager, deps: &[&str], documents: &[Value]) -> RetrievalManager {
    for dep in deps {
        m = match *dep {
            "embedder" => m.with_embedder(fake_embedder()),
            "llm" => m.with_llm(Arc::new(fake_llm)),
            "query_generator" => m.with_query_generator(Arc::new(query_generator)),
            "reranker" => m.with_reranker(Arc::new(reranker)),
            "compressor" => m.with_compressor(Arc::new(compressor)),
            "query_constructor" => m.with_query_constructor(query_constructor),
            "neighbor_loader" => m.with_neighbor_loader(neighbor_loader),
            "router" => m.with_router(router),
            "provider" => m.with_provider(Arc::new(reversed_corpus(documents))),
            "custom" => m.with_custom(Arc::new(reversed_corpus(documents))),
            other => panic!("dependência desconhecida: {other}"),
        };
    }
    m
}
