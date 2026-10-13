//! Configuração compacta e catálogo público de estratégias — o equivalente de `chunk_config.py`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{ChunkError, Result};

/// As 25 estratégias do chatbot (mesma ordem do `SUPPORTED_STRATEGIES` do Python) seguidas das
/// 8 estratégias estruturais/contextuais acrescentadas no port.
pub const SUPPORTED_STRATEGIES: &[&str] = &[
    "character",
    "recursive",
    "word",
    "sentence",
    "paragraph",
    "line",
    "token",
    "token_tiktoken",
    "token_sentence_transformer",
    "semchunk",
    "semantic",
    "nltk",
    "spacy",
    "markdown",
    "markdown_header",
    "markdown_syntax",
    "html",
    "html_header",
    "html_section",
    "html_semantic",
    "json",
    "latex",
    "code",
    "python",
    "javascript",
    // Novas
    "markdown_header_contextual",
    "docling_hybrid",
    "table",
    "parent_child",
    "contextual_retrieval",
    "legal",
    "qa",
    "late_chunking",
];

/// Estratégias que precisam de modelo (tokenizador, embeddings, LLM) carregado ou injetado.
pub const MODEL_STRATEGIES: &[&str] =
    &["token", "semchunk", "semantic", "docling_hybrid", "contextual_retrieval", "late_chunking"];

/// Apelidos aceitos (os do Python + os das estratégias novas).
pub const STRATEGY_ALIASES: &[(&str, &str)] = &[
    ("char", "character"),
    ("fixed", "character"),
    ("fixed_size", "character"),
    ("sliding_window", "character"),
    ("delimiter", "character"),
    ("separator", "character"),
    ("regex", "character"),
    ("recursive_character", "recursive"),
    ("huggingface_token", "token"),
    ("token_huggingface", "token"),
    ("tokenizer", "token"),
    ("tiktoken", "token_tiktoken"),
    ("openai_token", "token_tiktoken"),
    ("sentence_transformers_token", "token_sentence_transformer"),
    ("semantic_token", "semchunk"),
    ("sentences", "sentence"),
    ("words", "word"),
    ("paragraphs", "paragraph"),
    ("lines", "line"),
    ("markdown_headers", "markdown_header"),
    ("markdown_syntax_aware", "markdown_syntax"),
    ("html_headers", "html_header"),
    ("html_sections", "html_section"),
    ("recursive_json", "json"),
    ("code_language", "code"),
    ("python_code", "python"),
    ("js", "javascript"),
    ("javascript_framework", "javascript"),
    ("semantic_embedding", "semantic"),
    // Novas
    ("contextual_markdown_header", "markdown_header_contextual"),
    ("markdown_contextual", "markdown_header_contextual"),
    ("contextual_header", "markdown_header_contextual"),
    ("hybrid", "docling_hybrid"),
    ("docling", "docling_hybrid"),
    ("table_aware", "table"),
    ("tables", "table"),
    ("small_to_big", "parent_child"),
    ("parent_document", "parent_child"),
    ("contextual", "contextual_retrieval"),
    ("anthropic_contextual", "contextual_retrieval"),
    ("legislation", "legal"),
    ("legislacao", "legal"),
    ("lei", "legal"),
    ("faq", "qa"),
    ("question_answer", "qa"),
    ("late", "late_chunking"),
];

const BREAKPOINT_TYPES: &[&str] = &["percentile", "standard_deviation", "gradient", "interquartile"];

/// Nome canônico e validado da estratégia.
pub fn normalize_strategy(strategy: &str) -> Result<&'static str> {
    if strategy.trim().is_empty() {
        return Err(ChunkError::config("strategy must be a non-empty string"));
    }
    let normalized = strategy.trim().to_lowercase().replace(['-', ' '], "_");
    let canonical = STRATEGY_ALIASES.iter().find(|(a, _)| *a == normalized).map_or(normalized.as_str(), |(_, c)| *c);
    SUPPORTED_STRATEGIES
        .iter()
        .find(|s| **s == canonical)
        .copied()
        .ok_or_else(|| ChunkError::UnsupportedStrategy(strategy.to_string(), SUPPORTED_STRATEGIES.join(", ")))
}

pub fn requires_model(strategy: &str) -> Result<bool> {
    Ok(MODEL_STRATEGIES.contains(&normalize_strategy(strategy)?))
}

/// Toda a configuração de qualquer estratégia. Na maioria dos casos bastam `strategy`,
/// `chunk_size` e `chunk_overlap`; o resto é avançado, e argumentos específicos de cada
/// estratégia ficam em `options`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkConfig {
    #[serde(default = "default_strategy")]
    pub strategy: String,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default)]
    pub chunk_overlap: usize,
    /// Tokenizador/modelo de embeddings: id do Hugging Face Hub ou caminho local.
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub document_type: Option<String>,
    #[serde(default = "default_breakpoint")]
    pub breakpoint_threshold_type: String,
    #[serde(default)]
    pub options: Map<String, Value>,
}

fn default_strategy() -> String {
    "recursive".into()
}
fn default_chunk_size() -> usize {
    1000
}
fn default_model_name() -> String {
    // Mesmo padrão do `ChunkConfig` do Python.
    "BAAI/bge-small-zh-v1.5".into()
}
fn default_breakpoint() -> String {
    "gradient".into()
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            strategy: default_strategy(),
            chunk_size: default_chunk_size(),
            chunk_overlap: 0,
            model_name: default_model_name(),
            language: None,
            document_type: None,
            breakpoint_threshold_type: default_breakpoint(),
            options: Map::new(),
        }
    }
}

impl ChunkConfig {
    pub fn new(strategy: &str, chunk_size: usize, chunk_overlap: usize) -> Result<Self> {
        Self { strategy: strategy.to_string(), chunk_size, chunk_overlap, ..Self::default() }.validated()
    }

    /// Monta a partir de um mapa JSON (o formato do YAML do chatbot), aceitando os nomes
    /// legados `type_split_code` e `type_document`.
    pub fn from_value(value: Value) -> Result<Self> {
        let mut map = match value {
            Value::Null => Map::new(),
            Value::Object(m) => m,
            other => return Err(ChunkError::config(format!("chunk configuration must be a mapping, got {other}"))),
        };
        for (old, new) in [("type_split_code", "language"), ("type_document", "document_type")] {
            if map.contains_key(old)
                && !map.contains_key(new)
                && let Some(v) = map.remove(old)
            {
                map.insert(new.to_string(), v);
            }
        }
        const KNOWN: &[&str] = &[
            "strategy",
            "chunk_size",
            "chunk_overlap",
            "model_name",
            "language",
            "document_type",
            "breakpoint_threshold_type",
            "options",
        ];
        let mut unknown: Vec<&String> = map.keys().filter(|k| !KNOWN.contains(&k.as_str())).collect();
        if !unknown.is_empty() {
            unknown.sort();
            let names: Vec<&str> = unknown.iter().map(|s| s.as_str()).collect();
            return Err(ChunkError::config(format!("Unknown chunk configuration option(s): {}", names.join(", "))));
        }
        for key in ["chunk_size", "chunk_overlap"] {
            if let Some(v) = map.get(key)
                && !(v.is_u64() || v.is_i64())
            {
                return Err(ChunkError::config(if key == "chunk_size" {
                    "chunk_size must be a positive integer"
                } else {
                    "chunk_overlap must be an integer between 0 and chunk_size - 1"
                }));
            }
            if let Some(v) = map.get(key)
                && v.as_i64().is_some_and(|n| n < 0)
            {
                return Err(ChunkError::config(format!("{key} must be non-negative")));
            }
        }
        let cfg: ChunkConfig = serde_json::from_value(Value::Object(map)).map_err(|e| ChunkError::config(e.to_string()))?;
        cfg.validated()
    }

    fn validated(mut self) -> Result<Self> {
        self.strategy = normalize_strategy(&self.strategy)?.to_string();
        if self.chunk_size < 1 {
            return Err(ChunkError::config("chunk_size must be a positive integer"));
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(ChunkError::config("chunk_overlap must be an integer between 0 and chunk_size - 1"));
        }
        if !BREAKPOINT_TYPES.contains(&self.breakpoint_threshold_type.as_str()) {
            return Err(ChunkError::config(
                "breakpoint_threshold_type must be percentile, standard_deviation, gradient or interquartile",
            ));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn aliases_and_validation() {
        assert_eq!(normalize_strategy(" Sliding-Window ").unwrap(), "character");
        assert_eq!(normalize_strategy("faq").unwrap(), "qa");
        assert!(normalize_strategy("nope").is_err());
        assert_eq!(SUPPORTED_STRATEGIES.len(), 33);
    }

    #[test]
    fn from_value_like_python() {
        let c = ChunkConfig::from_value(json!({"strategy": "code", "type_split_code": "go", "chunk_size": 10})).unwrap();
        assert_eq!(c.language.as_deref(), Some("go"));
        assert!(ChunkConfig::from_value(json!({"strategy": "x"})).is_err());
        assert!(ChunkConfig::from_value(json!({"chunk_size": 10, "chunk_overlap": 10})).is_err());
        assert!(ChunkConfig::from_value(json!({"chunk_size": true})).is_err());
        let err = ChunkConfig::from_value(json!({"foo": 1, "bar": 2})).unwrap_err().to_string();
        assert!(err.contains("bar, foo"), "{err}");
    }
}
