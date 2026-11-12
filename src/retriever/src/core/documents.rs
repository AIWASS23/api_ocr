//! Documentos: conversão de valores soltos (`as_document` do Python), chave de deduplicação,
//! nota e anotação de posição/estratégia.

use chunking::{Document, Metadata};
use serde_json::{Map, Value};

/// `str(valor)` do Python para os tipos do JSON.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `as_document` do Python: `Document`/mapa (`page_content`, `text`, `content`, `document` ou
/// `data`), par `[documento, nota]` ou qualquer valor como texto.
pub fn as_document(value: Value) -> Document {
    match value {
        Value::Object(m) => {
            let content = ["page_content", "text", "content", "document", "data"]
                .iter()
                .find_map(|k| m.get(*k).filter(|v| truthy(v)))
                .map(py_str)
                .unwrap_or_default();
            let mut metadata: Metadata = ["metadata", "metadatas"]
                .iter()
                .find_map(|k| m.get(*k).filter(|v| truthy(v)))
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            for key in ["id", "source", "url", "score", "distance", "title"] {
                if let Some(v) = m.get(key)
                    && !metadata.contains_key(key)
                {
                    metadata.insert(key.into(), v.clone());
                }
            }
            Document::with_metadata(content, metadata)
        }
        Value::Array(items) if !items.is_empty() => {
            let score = items.get(1).and_then(Value::as_f64);
            let mut doc = as_document(items.into_iter().next().unwrap_or(Value::Null));
            if let Some(s) = score {
                doc.metadata.insert("score".into(), Value::from(s));
            }
            doc
        }
        other => Document::new(py_str(&other)),
    }
}

/// Chave de deduplicação na fusão (`document_key` do Python).
pub fn document_key(doc: &Document) -> String {
    for key in ["id", "pk", "document_id", "doc_id", "source", "url"] {
        if let Some(v) = doc.metadata.get(key).filter(|v| !v.is_null()) {
            return format!("{key}:{}", py_str(v));
        }
    }
    format!("content:{}", doc.page_content)
}

/// Cópia com `metadata.retrieval_score`.
pub fn with_score(doc: &Document, score: f64) -> Document {
    let mut d = doc.clone();
    d.metadata.insert("retrieval_score".into(), Value::from(score));
    d
}

/// `_annotate`: os `k` primeiros, com `retrieval_rank` e (se ainda não tiver)
/// `retrieval_strategy`.
pub fn annotate(values: Vec<Document>, strategy: &str, k: usize) -> Vec<Document> {
    values
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(i, mut d)| {
            d.metadata.entry("retrieval_strategy").or_insert_with(|| Value::String(strategy.into()));
            d.metadata.insert("retrieval_rank".into(), Value::from(i + 1));
            d
        })
        .collect()
}

/// Nota de um documento (0 quando não tem).
pub fn score_of(doc: &Document) -> f64 {
    doc.metadata.get("retrieval_score").and_then(Value::as_f64).unwrap_or(0.0)
}

/// Documento a partir de um mapa JSON qualquer (útil para `parent_store`).
pub fn from_map(m: &Map<String, Value>) -> Document {
    as_document(Value::Object(m.clone()))
}
