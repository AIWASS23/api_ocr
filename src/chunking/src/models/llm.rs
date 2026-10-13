//! Modelo que escreve o contexto de cada chunk (estratégia `contextual_retrieval`).
//! [`OpenAiCompatible`] (feature `llm`) fala com qualquer servidor `/v1/chat/completions`
//! (Ollama, vLLM, llama.cpp server, OpenAI...), o mesmo papel do `LLMManager` do chatbot.

use crate::error::Result;

pub trait ContextModel: Send + Sync {
    /// Uma ou duas frases que situam `chunk` dentro de `document`.
    fn situate(&self, document: &str, chunk: &str) -> Result<String>;
}

/// Prompt do *Contextual Retrieval* (Anthropic), em português.
pub const DEFAULT_CONTEXT_PROMPT: &str = "<document>\n{document}\n</document>\n\
Aqui está o trecho que queremos situar dentro do documento completo:\n<chunk>\n{chunk}\n</chunk>\n\
Escreva um contexto curto e sucinto que situe este trecho no documento completo, para melhorar \
a busca do trecho. Responda apenas com o contexto, no mesmo idioma do documento.";

pub fn render_prompt(template: &str, document: &str, chunk: &str) -> String {
    template.replace("{document}", document).replace("{chunk}", chunk)
}

#[cfg(feature = "llm")]
pub use http::OpenAiCompatible;

#[cfg(feature = "llm")]
mod http {
    use std::time::Duration;

    use serde_json::{Value, json};

    use super::{ContextModel, DEFAULT_CONTEXT_PROMPT, render_prompt};
    use crate::error::{ChunkError, Result};

    pub struct OpenAiCompatible {
        client: reqwest::blocking::Client,
        base_url: String,
        model: String,
        api_key: Option<String>,
        pub prompt: String,
        pub temperature: f64,
        pub max_tokens: u32,
        /// Campos extras do corpo da requisição (ex.: `{"reasoning_effort": "none"}` para
        /// desligar o raciocínio de modelos como qwen3 no Ollama).
        pub extra: serde_json::Map<String, Value>,
    }

    impl OpenAiCompatible {
        /// `base_url` termina em `/v1` (ex.: `http://localhost:11434/v1` para o Ollama).
        pub fn new(base_url: &str, model: &str, api_key: Option<String>, timeout: Duration) -> Result<Self> {
            let client = reqwest::blocking::Client::builder()
                .timeout(timeout)
                .build()
                .map_err(|e| ChunkError::Llm(e.to_string()))?;
            Ok(Self {
                client,
                base_url: base_url.trim_end_matches('/').to_string(),
                model: model.to_string(),
                api_key,
                prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
                temperature: 0.0,
                max_tokens: 200,
                extra: serde_json::Map::new(),
            })
        }
    }

    impl ContextModel for OpenAiCompatible {
        fn situate(&self, document: &str, chunk: &str) -> Result<String> {
            let mut body = json!({
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": render_prompt(&self.prompt, document, chunk)}],
            });
            if let Value::Object(b) = &mut body {
                b.extend(self.extra.clone());
            }
            let mut req = self.client.post(format!("{}/chat/completions", self.base_url)).json(&body);
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            let resp = req.send().map_err(|e| ChunkError::Llm(e.to_string()))?;
            let status = resp.status();
            let value: Value = resp.json().map_err(|e| ChunkError::Llm(e.to_string()))?;
            if !status.is_success() {
                return Err(ChunkError::Llm(format!("HTTP {status}: {value}")));
            }
            let content = value["choices"][0]["message"]["content"].as_str().map(str::trim).unwrap_or("");
            if content.is_empty() {
                // Modelos de raciocínio podem gastar todos os tokens "pensando".
                return Err(ChunkError::Llm(format!(
                    "resposta vazia (finish_reason={}); para modelos de raciocínio, aumente llm_max_tokens \
                     ou desligue o raciocínio (ex.: llm_extra: {{\"reasoning_effort\": \"none\"}})",
                    value["choices"][0]["finish_reason"]
                )));
            }
            Ok(content.to_string())
        }
    }
}
