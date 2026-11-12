//! Filtro por metadado (`_apply_filters` do Python): igualdade, ou pertencer à lista.

use chunking::Document;
use serde_json::{Map, Value};

/// `==` do Python entre valores do JSON (1 == 1.0).
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

pub fn matches(doc: &Document, filters: &Map<String, Value>) -> bool {
    filters.iter().all(|(key, expected)| {
        let actual = doc.metadata.get(key).unwrap_or(&Value::Null);
        match expected {
            Value::Array(options) => options.iter().any(|o| py_eq(actual, o)),
            other => py_eq(actual, other),
        }
    })
}

pub fn apply(docs: Vec<Document>, filters: &Map<String, Value>) -> Vec<Document> {
    if filters.is_empty() {
        return docs;
    }
    docs.into_iter().filter(|d| matches(d, filters)).collect()
}
