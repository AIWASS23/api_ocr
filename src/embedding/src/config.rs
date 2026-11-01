//! Configuração compacta e catálogo de estratégias — o equivalente de `embedding_config.py`.
//!
//! As estratégias de provedores pagos do Python (`openai`, `azure_openai`, `cohere`, `voyage`,
//! `gemini`, `mistral`, `huggingface_inference`) ficam de fora: o nome é reconhecido (com os
//! apelidos) e recusado com [`EmbeddingError::PaidProvider`]. O `ollama`, que no Python mora em
//! `paid/`, roda localmente e de graça: entra.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{EmbeddingError, Result};

/// Estratégias deste crate: as locais e gratuitas do Python, na mesma ordem, e as estratégias
/// só de lógica acrescentadas no port (`bm25`, `lsa`).
pub const SUPPORTED_STRATEGIES: &[&str] = &[
    // Densas locais
    "sentence_transformer",
    "sbert",
    "bge",
    "gte",
    "e5",
    "multilingual",
    "instructor",
    "jina",
    "nomic",
    "huggingface",
    // Esparsas locais
    "sparse",
    "tfidf",
    "hashing",
    // Multimodais
    "clip",
    "siglip",
    // Servidor local
    "ollama",
    // Extensão (embedder injetado)
    "custom",
    // Novas: só lógica, sem modelo
    "bm25",
    "lsa",
];

/// Provedores pagos do Python, recusados com uma mensagem própria.
pub const PAID_STRATEGIES: &[&str] =
    &["openai", "azure_openai", "cohere", "voyage", "gemini", "mistral", "huggingface_inference"];

/// Apelidos do Python (inclusive os dos provedores pagos, para a mensagem certa) e os das novas.
pub const STRATEGY_ALIASES: &[(&str, &str)] = &[
    ("sentence_transformers", "sentence_transformer"),
    ("st", "sentence_transformer"),
    ("bert", "huggingface"),
    ("transformers", "huggingface"),
    ("hf", "huggingface"),
    ("e5_multilingual", "e5"),
    ("multilingual_e5", "e5"),
    ("instructor_embedding", "instructor"),
    ("jina_embedding", "jina"),
    ("nomic_embedding", "nomic"),
    ("splade", "sparse"),
    ("sparse_encoder", "sparse"),
    ("sentence_transformer_sparse", "sparse"),
    ("tf_idf", "tfidf"),
    ("feature_hashing", "hashing"),
    ("openai_clip", "clip"),
    ("hf_clip", "clip"),
    ("azure", "azure_openai"),
    ("azure_embeddings", "azure_openai"),
    ("vertex", "gemini"),
    ("vertex_ai", "gemini"),
    ("google", "gemini"),
    ("google_genai", "gemini"),
    ("huggingface_api", "huggingface_inference"),
    ("hf_inference", "huggingface_inference"),
    ("local_api", "ollama"),
    // Novas
    ("okapi_bm25", "bm25"),
    ("bm25_sparse", "bm25"),
    ("lsi", "lsa"),
    ("latent_semantic", "lsa"),
    ("latent_semantic_analysis", "lsa"),
    ("tfidf_svd", "lsa"),
    ("truncated_svd", "lsa"),
];

/// Modelo padrão de cada estratégia (`None`: não usa modelo). Os mesmos do Python, exceto o
/// `sparse`: o `naver/splade-v3` é restrito (licença não comercial, acesso por aprovação) e não
/// tem ONNX; o padrão aqui é o `prithivida/Splade_PP_en_v1` (Apache-2.0).
pub fn default_model(strategy: &str) -> Option<&'static str> {
    Some(match strategy {
        "sentence_transformer" | "sbert" | "huggingface" => "sentence-transformers/all-MiniLM-L6-v2",
        "bge" => "BAAI/bge-m3",
        "gte" => "Alibaba-NLP/gte-multilingual-base",
        "e5" => "intfloat/multilingual-e5-base",
        "multilingual" => "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "instructor" => "hkunlp/instructor-large",
        "jina" => "jinaai/jina-embeddings-v3",
        "nomic" => "nomic-ai/nomic-embed-text-v1.5",
        "sparse" => "prithivida/Splade_PP_en_v1",
        "clip" => "openai/clip-vit-base-patch32",
        "siglip" => "google/siglip-base-patch16-224",
        "ollama" => "nomic-embed-text",
        _ => return None,
    })
}

/// Famílias densas carregadas pelo pipeline sentence-transformers.
pub const LOCAL_DENSE_STRATEGIES: &[&str] =
    &["sentence_transformer", "sbert", "bge", "gte", "e5", "multilingual", "instructor", "jina", "nomic"];

/// Estratégias que precisam de um modelo (local, baixado do Hub ou servidor).
pub const MODEL_STRATEGIES: &[&str] = &[
    "sentence_transformer",
    "sbert",
    "bge",
    "gte",
    "e5",
    "multilingual",
    "instructor",
    "jina",
    "nomic",
    "huggingface",
    "sparse",
    "clip",
    "siglip",
    "ollama",
];

/// Estratégias que aprendem o vocabulário no primeiro lote de documentos.
pub const FITTED_STRATEGIES: &[&str] = &["tfidf", "bm25", "lsa"];

/// Nome canônico e validado da estratégia.
pub fn normalize_strategy(strategy: &str) -> Result<&'static str> {
    if strategy.trim().is_empty() {
        return Err(EmbeddingError::config("strategy must be a non-empty string"));
    }
    let normalized = strategy.trim().to_lowercase().replace(['-', ' '], "_");
    let canonical = STRATEGY_ALIASES.iter().find(|(a, _)| *a == normalized).map_or(normalized.as_str(), |(_, c)| *c);
    if let Some(paid) = PAID_STRATEGIES.iter().find(|p| **p == canonical) {
        return Err(EmbeddingError::PaidProvider(paid.to_string()));
    }
    SUPPORTED_STRATEGIES
        .iter()
        .find(|s| **s == canonical)
        .copied()
        .ok_or_else(|| EmbeddingError::UnsupportedStrategy(strategy.to_string(), SUPPORTED_STRATEGIES.join(", ")))
}

/// Configuração completa. Normalmente bastam `strategy` e, opcionalmente, `model_name`; ajustes
/// específicos de cada estratégia ficam em `options`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    #[serde(default = "default_strategy")]
    pub strategy: String,
    /// Id do Hugging Face Hub ou diretório local. Padrão: o da estratégia.
    #[serde(default)]
    pub model_name: Option<String>,
    /// Usado só pelo `ollama` atrás de um proxy autenticado (`Authorization: Bearer`). Aceito
    /// também para ler configurações do Python sem erro.
    #[serde(default)]
    pub api_key: Option<String>,
    /// `cpu` (padrão), `cuda`/`cuda:N`/`gpu` ou `auto` (provedor de execução do ONNX Runtime
    /// escolhido por `DOCLING_RS_EP`).
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Normaliza cada vetor para norma 1 (depois do corte de `dimensions`).
    #[serde(default)]
    pub normalize: bool,
    /// Corta os vetores nas primeiras `dimensions` posições (Matryoshka) e, nas estratégias
    /// esparsas, limita o vocabulário.
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

fn default_strategy() -> String {
    "sentence_transformer".into()
}
fn default_device() -> String {
    "cpu".into()
}
fn default_batch_size() -> usize {
    32
}

// Nunca imprime o `api_key`.
impl std::fmt::Debug for EmbeddingConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingConfig")
            .field("strategy", &self.strategy)
            .field("model_name", &self.model_name)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("device", &self.device)
            .field("batch_size", &self.batch_size)
            .field("normalize", &self.normalize)
            .field("dimensions", &self.dimensions)
            .field("options", &self.options)
            .finish()
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            strategy: default_strategy(),
            model_name: None,
            api_key: None,
            device: default_device(),
            batch_size: default_batch_size(),
            normalize: false,
            dimensions: None,
            options: Map::new(),
        }
        .validated()
        .expect("configuração padrão válida")
    }
}

impl EmbeddingConfig {
    /// Estratégia com o modelo padrão dela.
    pub fn new(strategy: &str) -> Result<Self> {
        Self { strategy: strategy.to_string(), model_name: None, ..Self::default() }.validated()
    }

    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    /// A partir de um mapa JSON (o `embedding:` do YAML do chatbot), aceitando os nomes legados
    /// `embedding_strategy`, `embedding_model_name` e `embedding_token`.
    pub fn from_value(value: Value) -> Result<Self> {
        let mut map = match value {
            Value::Null => Map::new(),
            Value::Object(m) => m,
            other => return Err(EmbeddingError::config(format!("embedding configuration must be a mapping, got {other}"))),
        };
        for (old, new) in [("embedding_strategy", "strategy"), ("embedding_model_name", "model_name"), ("embedding_token", "api_key")] {
            if map.contains_key(old)
                && !map.contains_key(new)
                && let Some(v) = map.remove(old)
            {
                map.insert(new.to_string(), v);
            }
        }
        const KNOWN: &[&str] = &["strategy", "model_name", "api_key", "device", "batch_size", "normalize", "dimensions", "options"];
        let mut unknown: Vec<&str> = map.keys().map(String::as_str).filter(|k| !KNOWN.contains(k)).collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(EmbeddingError::config(format!("Unknown embedding configuration option(s): {}", unknown.join(", "))));
        }
        let positive = |key: &str, msg: &str| -> Result<()> {
            match map.get(key) {
                None | Some(Value::Null) if key == "dimensions" => Ok(()),
                None => Ok(()),
                Some(v) if v.as_u64().is_some_and(|n| n >= 1) => Ok(()),
                Some(_) => Err(EmbeddingError::config(msg)),
            }
        };
        positive("batch_size", "batch_size must be a positive integer")?;
        positive("dimensions", "dimensions must be a positive integer or None")?;
        if map.get("normalize").is_some_and(|v| !v.is_boolean()) {
            return Err(EmbeddingError::config("normalize must be a boolean"));
        }
        let cfg: EmbeddingConfig = serde_json::from_value(Value::Object(map)).map_err(|e| EmbeddingError::config(e.to_string()))?;
        cfg.validated()
    }

    /// Valida, troca apelidos pelo nome canônico e preenche o modelo padrão.
    pub fn validated(mut self) -> Result<Self> {
        self.strategy = normalize_strategy(&self.strategy)?.to_string();
        if self.model_name.as_deref().is_none_or(|m| m.trim().is_empty()) {
            self.model_name = default_model(&self.strategy).map(str::to_string);
        }
        if self.batch_size < 1 {
            return Err(EmbeddingError::config("batch_size must be a positive integer"));
        }
        if self.dimensions == Some(0) {
            return Err(EmbeddingError::config("dimensions must be a positive integer or None"));
        }
        Ok(self)
    }

    /// Nome do modelo, ou o da estratégia quando ela não usa modelo.
    pub fn model_or_strategy(&self) -> &str {
        self.model_name.as_deref().unwrap_or(&self.strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn alias_and_default_model() {
        let c = EmbeddingConfig::new("sentence-transformers").unwrap();
        assert_eq!(c.strategy, "sentence_transformer");
        assert_eq!(c.model_name.as_deref(), Some("sentence-transformers/all-MiniLM-L6-v2"));
        assert_eq!(EmbeddingConfig::new("LSI").unwrap().strategy, "lsa");
        assert!(EmbeddingConfig::new("tfidf").unwrap().model_name.is_none());
    }

    #[test]
    fn compact_legacy_mapping() {
        let c = EmbeddingConfig::from_value(json!({
            "embedding_strategy": "e5-multilingual",
            "embedding_model_name": "local/model",
            "embedding_token": "token",
        }))
        .unwrap();
        assert_eq!((c.strategy.as_str(), c.model_name.as_deref(), c.api_key.as_deref()), ("e5", Some("local/model"), Some("token")));
        assert!(!format!("{c:?}").contains("token\""), "api_key não aparece no Debug");
    }

    #[test]
    fn validation() {
        assert!(EmbeddingConfig::from_value(json!({"batch_size": 0})).is_err());
        assert!(EmbeddingConfig::from_value(json!({"dimensions": 0})).is_err());
        assert!(EmbeddingConfig::from_value(json!({"normalize": "yes"})).is_err());
        assert!(EmbeddingConfig::from_value(json!({"dimensions": null})).is_ok());
        assert!(matches!(EmbeddingConfig::new("unknown"), Err(EmbeddingError::UnsupportedStrategy(..))));
        let err = EmbeddingConfig::from_value(json!({"zeta": 1, "alpha": 2})).unwrap_err().to_string();
        assert!(err.ends_with("alpha, zeta"), "{err}");
    }

    #[test]
    fn paid_providers_are_refused_by_name_and_alias() {
        for name in ["openai", "azure", "cohere", "voyage", "vertex_ai", "mistral", "hf_inference"] {
            assert!(matches!(normalize_strategy(name), Err(EmbeddingError::PaidProvider(_))), "{name}");
        }
        assert_eq!(normalize_strategy("local_api").unwrap(), "ollama");
    }
}
