//! Cliente de LLM compatível com OpenAI (`/chat/completions`): Ollama, vLLM, llama.cpp,
//! LM Studio... Configurado por `llm` na configuração (feature `llm`):
//! `{"model": "...", "base_url": "http://localhost:11434/v1", "api_key_env": "NOME_DA_VARIAVEL",
//! "timeout_secs": 60, "max_tokens": 512, "temperature": 0, "extra": {...}}`. Sem `api_key_env`
//! (ou `api_key`), nenhuma chave é enviada.

use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::deps::Llm;
use crate::error::{RetrievalError, Result};

pub struct OpenAiCompatible {
    client: reqwest::blocking::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    max_tokens: Option<u64>,
    temperature: f64,
    extra: Map<String, Value>,
}

impl OpenAiCompatible {
    pub fn from_config(cfg: &Value) -> Result<Self> {
        let Value::Object(m) = cfg else { return Err(RetrievalError::config("llm must be a mapping")) };
        let s = |k: &str| m.get(k).and_then(Value::as_str).map(str::to_string);
        let model = s("model").ok_or_else(|| RetrievalError::config("llm.model is required"))?;
        let base = s("base_url").unwrap_or_else(|| "http://localhost:11434/v1".into());
        // A chave só sai de uma variável nomeada na configuração: nunca uma variável padrão
        // enviada a um `base_url` qualquer.
        let api_key = s("api_key").or_else(|| s("api_key_env").and_then(|name| std::env::var(name).ok()));
        let timeout = m.get("timeout_secs").and_then(Value::as_f64).unwrap_or(60.0);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs_f64(timeout.max(1.0)))
            .build()
            .map_err(RetrievalError::model)?;
        Ok(Self {
            client,
            url: format!("{}/chat/completions", base.trim_end_matches('/')),
            model,
            api_key,
            max_tokens: m.get("max_tokens").and_then(Value::as_u64),
            temperature: m.get("temperature").and_then(Value::as_f64).unwrap_or(0.0),
            extra: m.get("extra").and_then(Value::as_object).cloned().unwrap_or_default(),
        })
    }
}

impl Llm for OpenAiCompatible {
    fn complete(&self, prompt: &str) -> Result<String> {
        let mut body = json!({"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": self.temperature});
        if let Some(n) = self.max_tokens {
            body["max_tokens"] = json!(n);
        }
        if let Value::Object(b) = &mut body {
            b.extend(self.extra.clone());
        }
        let mut req = self.client.post(&self.url).json(&body);
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().map_err(|e| RetrievalError::model(format!("llm {}: {e}", self.url)))?;
        let status = resp.status();
        let v: Value = resp.json().map_err(|e| RetrievalError::model(format!("llm: invalid response ({status}): {e}")))?;
        if !status.is_success() {
            return Err(RetrievalError::model(format!("llm {status}: {}", v.get("error").map(Value::to_string).unwrap_or_default())));
        }
        let message = v.pointer("/choices/0/message");
        let content = message.and_then(|m| m.get("content")).and_then(Value::as_str).unwrap_or("");
        // Resposta vazia é válida (na compressão, "nada relevante"); só o caso de um modelo de
        // raciocínio que gastou tudo pensando vira erro.
        let reasoned = ["reasoning_content", "reasoning"]
            .iter()
            .any(|k| message.and_then(|m| m.get(*k)).and_then(Value::as_str).is_some_and(|r| !r.trim().is_empty()));
        if content.trim().is_empty() && reasoned {
            return Err(RetrievalError::model(
                "llm returned only reasoning, no content (try llm.extra = {\"reasoning_effort\": \"none\"} or a larger max_tokens)",
            ));
        }
        Ok(content.to_string())
    }
}
