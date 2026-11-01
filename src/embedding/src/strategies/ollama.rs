//! `ollama` — servidor Ollama local (`OllamaEmbedder` do chatbot): `POST /api/embed` com o
//! modelo e os textos. Gratuito e local (por isso entra, apesar de estar em `paid/` no Python).
//!
//! Endereço: `options.host`, `options.client_options.host`, a variável `OLLAMA_HOST` ou
//! `http://localhost:11434`. `api_key`, se houver, vai como `Authorization: Bearer` (Ollama
//! atrás de um proxy autenticado).
//!
//! Padrão: `nomic-embed-text` (`ollama pull nomic-embed-text`). Opções: `host`, `timeout_secs`
//! (60), `client_options` (`host`, `timeout`), `embed_options` (vão no corpo da requisição:
//! `truncate`, `keep_alive`, `dimensions`, `options`).

use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::config::EmbeddingConfig;
use crate::core::options::Options;
use crate::embedder::{Embedder, InputType};
use crate::error::{EmbeddingError, Result};

pub struct OllamaEmbedder {
    client: reqwest::blocking::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    extra: Map<String, Value>,
    batch_size: usize,
}

/// `http://host:porta`, aceitando `host`, `host:porta` e `0.0.0.0` como no cliente Python.
pub fn normalize_host(raw: &str) -> String {
    let raw = raw.trim().trim_end_matches('/');
    let (scheme, rest) = match raw.split_once("://") {
        Some((s, r)) => (s.to_string(), r.to_string()),
        None => ("http".to_string(), raw.to_string()),
    };
    let rest = if rest.is_empty() { "localhost".to_string() } else { rest };
    let rest = rest.replacen("0.0.0.0", "localhost", 1);
    let has_port = rest.rsplit_once(':').is_some_and(|(_, p)| p.chars().all(|c| c.is_ascii_digit()) && !p.is_empty());
    if has_port { format!("{scheme}://{rest}") } else { format!("{scheme}://{rest}:11434") }
}

pub fn load(config: &EmbeddingConfig) -> Result<OllamaEmbedder> {
    let mut o = Options::new("ollama", &config.options);
    let mut client_options = Options::new("client_options", &o.object("client_options")?);
    let host = o
        .string("host")?
        .or(client_options.string("host")?)
        .or_else(|| std::env::var("OLLAMA_HOST").ok())
        .unwrap_or_else(|| "http://localhost:11434".into());
    let timeout = o.f64("timeout_secs")?.or(client_options.f64("timeout")?).unwrap_or(60.0);
    client_options.finish()?;
    let extra = o.object("embed_options")?;
    for k in ["model", "input"] {
        if extra.contains_key(k) {
            return Err(EmbeddingError::config(format!("ollama: embed_options cannot set '{k}'")));
        }
    }
    o.finish()?;
    let model = config.model_name.clone().ok_or_else(|| EmbeddingError::config("model_name is required for strategy 'ollama'"))?;
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs_f64(timeout.max(1.0)))
        .build()
        .map_err(|e| EmbeddingError::Http(e.to_string()))?;
    Ok(OllamaEmbedder { client, url: format!("{}/api/embed", normalize_host(&host)), model, api_key: config.api_key.clone(), extra, batch_size: config.batch_size })
}

impl Embedder for OllamaEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn embed(&self, texts: &[&str], _input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size.max(1)) {
            let mut body = json!({"model": self.model, "input": batch});
            if let Value::Object(m) = &mut body {
                m.extend(self.extra.clone());
            }
            let mut req = self.client.post(&self.url).json(&body);
            if let Some(k) = &self.api_key {
                req = req.bearer_auth(k);
            }
            let resp = req.send().map_err(|e| EmbeddingError::Http(format!("ollama {}: {e}", self.url)))?;
            let status = resp.status();
            let value: Value = resp.json().map_err(|e| EmbeddingError::Http(format!("ollama: invalid response ({status}): {e}")))?;
            if !status.is_success() {
                let msg = value.get("error").and_then(Value::as_str).unwrap_or("");
                return Err(EmbeddingError::Http(format!("ollama {status}: {msg}")));
            }
            let vectors: Vec<Vec<f32>> = serde_json::from_value(value.get("embeddings").cloned().unwrap_or(Value::Null))
                .map_err(|e| EmbeddingError::Http(format!("ollama: response without embeddings: {e}")))?;
            out.extend(vectors);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_host;

    #[test]
    fn hosts_like_the_python_client() {
        assert_eq!(normalize_host("localhost"), "http://localhost:11434");
        assert_eq!(normalize_host("0.0.0.0:11434"), "http://localhost:11434");
        assert_eq!(normalize_host("https://ollama.exemplo.com/"), "https://ollama.exemplo.com:11434");
        assert_eq!(normalize_host("http://10.0.0.5:8080"), "http://10.0.0.5:8080");
    }
}
