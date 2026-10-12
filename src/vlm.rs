use std::time::Duration;

use serde_json::{Value, json};

use crate::error::PrepError;

pub const DEFAULT_PROMPT: &str = "Describe the image in three sentences. Be consise and accurate.";

/// Cliente de um endpoint `/chat/completions` compatível com OpenAI
/// (Ollama, LM Studio, vLLM, serviços hospedados) usado para descrever figuras.
#[derive(Debug, Clone)]
pub struct VlmClient {
    http: reqwest::blocking::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    prompt: String,
}

impl VlmClient {
    pub fn new(
        url: impl Into<String>,
        model: impl Into<String>,
        api_key: Option<String>,
        prompt: impl Into<String>,
    ) -> Result<Self, PrepError> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        Ok(Self {
            http,
            url: url.into(),
            model: model.into(),
            api_key,
            prompt: prompt.into(),
        })
    }

    /// `image_data_uri`: `data:<mime>;base64,<...>`.
    pub fn describe(&self, image_data_uri: &str) -> Result<String, PrepError> {
        let body = json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": self.prompt },
                    { "type": "image_url", "image_url": { "url": image_data_uri } },
                ],
            }],
        });

        let mut req = self.http.post(&self.url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp: Value = req.send()?.error_for_status()?.json()?;
        resp.pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .ok_or_else(|| PrepError::Vlm(format!("resposta sem conteúdo: {resp}")))
    }
}
