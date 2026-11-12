//! Configuração compacta e catálogos — o equivalente de `retrieval_config.py`.
//!
//! Providers: os locais (`local`, `in_memory`, `vector_db`) e o `custom` (injeção) ficam. As
//! demais 114 entradas do catálogo do Python são integrações por SDK do LangChain (bancos
//! vetoriais, buscas gerenciadas, web): são reconhecidas pelo nome e apelidos e recusadas com
//! [`RetrievalError::ExternalProvider`], que indica como injetar um provider próprio.

use chunking::Document;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::core::documents::as_document;
use crate::error::{RetrievalError, Result};

/// As 44 estratégias, na ordem do Python.
pub const SUPPORTED_STRATEGIES: &[&str] = &[
    // Semântica/vetorial
    "dense",
    "dense_mmr",
    "similarity_threshold",
    "multimodal",
    // Esparsa/lexical
    "sparse",
    "bm25",
    "bm25_plus",
    "tfidf",
    "keyword",
    "boolean",
    "fuzzy",
    "metadata",
    // Fusão
    "hybrid",
    "weighted_hybrid",
    "rrf",
    "ensemble",
    // Transformação de consulta
    "multi_query",
    "query_expansion",
    "rephrase_query",
    "hyde",
    "step_back",
    "decomposition",
    "self_query",
    // Hierárquica/múltiplas representações
    "parent_document",
    "multi_vector",
    "small_to_big",
    // Compressão, ranking e tempo
    "contextual_compression",
    "rerank",
    "cross_encoder_rerank",
    "colbert_rerank",
    "llm_rerank",
    "diversity",
    "time_weighted",
    "recency",
    // Fontes especializadas
    "graph",
    "knowledge_graph",
    "sql",
    "web_search",
    "knowledge_base",
    // Orquestração/extensão
    "adaptive",
    "router",
    "fallback",
    "cascade",
    "custom",
];

pub const STRATEGY_ALIASES: &[(&str, &str)] = &[
    ("semantic", "dense"),
    ("vector", "dense"),
    ("similarity", "dense"),
    ("knn", "dense"),
    ("mmr", "dense_mmr"),
    ("max_marginal_relevance", "dense_mmr"),
    ("threshold", "similarity_threshold"),
    ("sparse_vector", "sparse"),
    ("bm25_okapi", "bm25"),
    ("tf_idf", "tfidf"),
    ("tf_idf_retrieval", "tfidf"),
    ("full_text", "keyword"),
    ("fts", "keyword"),
    ("filter", "metadata"),
    ("fusion", "rrf"),
    ("reciprocal_rank_fusion", "rrf"),
    ("weighted", "weighted_hybrid"),
    ("query_fusion", "multi_query"),
    ("expand_query", "query_expansion"),
    ("reprase", "rephrase_query"),
    ("reprase_query", "rephrase_query"),
    ("rephrase", "rephrase_query"),
    ("hypothetical_document", "hyde"),
    ("stepback", "step_back"),
    ("sub_queries", "decomposition"),
    ("metadata_query", "self_query"),
    ("parent", "parent_document"),
    ("small_to_big_retrieval", "small_to_big"),
    ("compression", "contextual_compression"),
    ("cross_encoder", "cross_encoder_rerank"),
    ("colbert", "colbert_rerank"),
    ("rankllm", "llm_rerank"),
    ("time", "time_weighted"),
    ("kg", "knowledge_graph"),
    ("web", "web_search"),
    ("kb", "knowledge_base"),
    ("dynamic", "adaptive"),
    ("failover", "fallback"),
    ("pipeline", "cascade"),
];

/// Providers deste crate.
pub const SUPPORTED_PROVIDERS: &[&str] = &["local", "vector_db", "in_memory", "custom"];

/// Integrações externas do catálogo do Python (reconhecidas e recusadas).
pub const EXTERNAL_PROVIDERS: &[&str] = &[
    "chroma", "qdrant", "milvus", "pinecone", "weaviate", "elasticsearch", "opensearch", "pgvector", "redis",
    "mongodb_atlas", "lancedb", "faiss", "annoy", "scann", "usearch", "vespa", "typesense", "meilisearch", "marqo",
    "neo4j", "supabase", "astradb", "cassandra", "singlestore", "clickhouse", "couchbase", "oracle_vector",
    "sqlserver_vector", "upstash_vector", "zilliz", "turbopuffer", "deep_lake", "alibaba_cloud_mysql",
    "azure_cosmos_mongo_vcore", "azure_cosmos_nosql", "azure_postgresql", "cockroachdb", "databricks", "ibm_db2",
    "amazon_documentdb", "gel", "alloydb", "bigquery_vector_search", "cloud_sql_mysql", "cloud_sql_postgresql",
    "firestore", "google_memorystore", "spanner", "bigtable", "vertex_ai_feature_store", "vertex_ai_vector_search",
    "kinetica", "lambdadb", "lindorm", "amazon_memorydb", "moorcheh", "oceanbase", "opengauss", "pgvector_store",
    "sap_hana", "surrealdb", "teradata", "valkey", "vdms", "vedb_mysql", "volcengine_mysql", "ydb", "zeusdb", "vectara",
    "azure_ai_search", "vertex_ai_search", "bedrock_kb", "amazon_kendra", "google_drive", "elasticsearch_retriever",
    "nvidia_rag", "tavily", "wikipedia", "arxiv", "pubmed", "you", "kay", "parallel_search", "perplexity_search", "exa",
    "linkup", "dappier", "perigon", "valyu", "zep", "metal", "remote", "chatgpt_plugin", "agentmail", "box", "cognee",
    "cohere_rag", "contextual_ai", "egnyte", "galaxia", "graph_rag", "greennode", "watsonx_retriever", "imap", "nebius",
    "nimble_extract", "nimble_search", "permit", "pinecone_rerank", "ragatouille", "sourcey", "spicedb", "vectorize",
    "zotero",
];

pub const PROVIDER_ALIASES: &[(&str, &str)] = &[
    ("internal", "vector_db"),
    ("vectordb", "vector_db"),
    ("vectorstore", "vector_db"),
    ("memory", "in_memory"),
    ("elastic", "elasticsearch"),
    ("es", "elasticsearch"),
    ("open_search", "opensearch"),
    ("postgres", "pgvector"),
    ("postgresql", "pgvector"),
    ("mongo", "mongodb_atlas"),
    ("mongodb", "mongodb_atlas"),
    ("deeplake", "deep_lake"),
    ("cosmos_nosql", "azure_cosmos_nosql"),
    ("cosmos_mongo", "azure_cosmos_mongo_vcore"),
    ("azure_search", "azure_ai_search"),
    ("vertex_search", "vertex_ai_search"),
    ("google_vertex_search", "vertex_ai_search"),
    ("aws_kb", "bedrock_kb"),
    ("amazon_kb", "bedrock_kb"),
    ("kendra", "amazon_kendra"),
    ("astra", "astradb"),
    ("astra_db", "astradb"),
    ("upstash", "upstash_vector"),
    ("oracle", "oracle_vector"),
    ("sqlserver", "sqlserver_vector"),
    ("mssql", "sqlserver_vector"),
    ("wiki", "wikipedia"),
    ("pub_med", "pubmed"),
    ("gdrive", "google_drive"),
    ("perplexity", "perplexity_search"),
    ("parallel", "parallel_search"),
    ("nvidia", "nvidia_rag"),
    ("watsonx", "watsonx_retriever"),
    ("http", "remote"),
];

/// Estratégias de fusão (não podem ser componentes de outra fusão).
pub const FUSION_STRATEGIES: &[&str] = &["hybrid", "weighted_hybrid", "rrf", "ensemble"];

/// Estratégias que não podem ser a base de outra (`base_strategy` cai para `dense`).
pub const NON_BASE_STRATEGIES: &[&str] = &[
    "multi_query",
    "query_expansion",
    "rephrase_query",
    "hyde",
    "step_back",
    "decomposition",
    "contextual_compression",
    "rerank",
    "cross_encoder_rerank",
    "colbert_rerank",
    "llm_rerank",
    "fallback",
    "cascade",
];

fn normal_form(value: &str) -> String {
    value.trim().to_lowercase().replace(['-', ' '], "_")
}

pub fn normalize_strategy(strategy: &str) -> Result<&'static str> {
    if strategy.trim().is_empty() {
        return Err(RetrievalError::config("strategy must be a non-empty string"));
    }
    let n = normal_form(strategy);
    let canonical = STRATEGY_ALIASES.iter().find(|(a, _)| *a == n).map_or(n.as_str(), |(_, c)| *c);
    SUPPORTED_STRATEGIES
        .iter()
        .find(|s| **s == canonical)
        .copied()
        .ok_or_else(|| RetrievalError::UnsupportedStrategy(strategy.to_string(), SUPPORTED_STRATEGIES.join(", ")))
}

pub fn normalize_provider(provider: &str) -> Result<&'static str> {
    if provider.trim().is_empty() {
        return Err(RetrievalError::config("provider must be a non-empty string"));
    }
    let n = normal_form(provider);
    let canonical = PROVIDER_ALIASES.iter().find(|(a, _)| *a == n).map_or(n.as_str(), |(_, c)| *c);
    if let Some(p) = SUPPORTED_PROVIDERS.iter().find(|p| **p == canonical) {
        return Ok(p);
    }
    if let Some(p) = EXTERNAL_PROVIDERS.iter().find(|p| **p == canonical) {
        return Err(RetrievalError::ExternalProvider(p.to_string()));
    }
    let mut all: Vec<&str> = SUPPORTED_PROVIDERS.to_vec();
    all.extend(EXTERNAL_PROVIDERS);
    Err(RetrievalError::UnsupportedProvider(provider.to_string(), all.join(", ")))
}

/// Toda a configuração de qualquer estratégia. Na maioria dos casos bastam `strategy`, `k` e os
/// documentos; parâmetros de cada estratégia ficam em `options`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub strategy: String,
    pub provider: String,
    pub k: usize,
    pub fetch_k: usize,
    pub top_n: usize,
    pub score_threshold: Option<f64>,
    pub collection_name: Option<String>,
    /// Acervo local (`local`, `in_memory`) ou base das estratégias lexicais.
    pub documents: Vec<Document>,
    /// Componentes da fusão (nomes de estratégia).
    pub retrievers: Vec<String>,
    pub weights: Vec<f64>,
    /// Filtro por metadado: igualdade, ou pertencer à lista.
    pub filters: Map<String, Value>,
    pub rerank_model: String,
    /// Configuração do crate `embedding` (`{"strategy": "e5", ...}`), quando o embedder não é
    /// injetado.
    pub embedding: Option<Value>,
    /// Cliente de LLM compatível com OpenAI (`{"base_url", "model", "api_key_env", ...}`), quando o
    /// LLM não é injetado (feature `llm`).
    pub llm: Option<Value>,
    pub options: Map<String, Value>,
    pub provider_options: Map<String, Value>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            strategy: "dense".into(),
            provider: "vector_db".into(),
            k: 5,
            fetch_k: 20,
            top_n: 3,
            score_threshold: None,
            collection_name: None,
            documents: Vec::new(),
            retrievers: Vec::new(),
            weights: Vec::new(),
            filters: Map::new(),
            rerank_model: "BAAI/bge-reranker-base".into(),
            embedding: None,
            llm: None,
            options: Map::new(),
            provider_options: Map::new(),
        }
    }
}

const KNOWN: &[&str] = &[
    "strategy",
    "provider",
    "k",
    "fetch_k",
    "top_n",
    "score_threshold",
    "collection_name",
    "documents",
    "retrievers",
    "weights",
    "filters",
    "rerank_model",
    "embedding",
    "llm",
    "options",
    "provider_options",
];

/// Chaves legadas do Python e o nome atual.
const LEGACY: &[(&str, &str)] = &[
    ("retrieval_strategy", "strategy"),
    ("retriever_provider", "provider"),
    ("k_neighbors", "k"),
    ("k_neighbors_rerank", "fetch_k"),
    ("top_n_rerank", "top_n"),
    ("embedding_manager", "embedding"),
    ("embedding_function", "embedding"),
    ("retrieval_funcs", "retrievers"),
    ("retriever_funcs", "retrievers"),
];

impl RetrievalConfig {
    /// Estratégia com os padrões.
    pub fn new(strategy: &str) -> Result<Self> {
        Self::from_value(Value::String(strategy.to_string()))
    }

    /// Como o `RetrievalConfig.from_value` do Python: aceita um nome de estratégia, um mapa ou
    /// nada; troca as chaves legadas; chaves desconhecidas vão para `options`.
    pub fn from_value(value: Value) -> Result<Self> {
        let mut data = match value {
            Value::Null => Map::new(),
            Value::String(s) => Map::from_iter([("strategy".to_string(), Value::String(s))]),
            Value::Object(m) => m,
            other => return Err(RetrievalError::config(format!("retrieval configuration must be a mapping, got {other}"))),
        };
        for (old, new) in LEGACY {
            if data.contains_key(*old)
                && !data.contains_key(*new)
                && let Some(v) = data.remove(*old)
            {
                data.insert(new.to_string(), v);
            }
        }
        if data.contains_key("db") {
            return Err(RetrievalError::config("'db' is a Python object; inject a SearchProvider with RetrievalManager::with_provider"));
        }
        let mut options = match data.remove("options") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m,
            Some(_) => return Err(RetrievalError::config("options must be a mapping")),
        };
        let unknown: Vec<String> = data.keys().filter(|k| !KNOWN.contains(&k.as_str())).cloned().collect();
        for k in unknown {
            if let Some(v) = data.remove(&k) {
                options.insert(k, v);
            }
        }

        let mut cfg = RetrievalConfig { options, ..Default::default() };
        let str_field = |data: &Map<String, Value>, k: &str| -> Result<Option<String>> {
            match data.get(k) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                Some(_) => Err(RetrievalError::config(format!("{k} must be a string"))),
            }
        };
        if let Some(s) = str_field(&data, "strategy")? {
            cfg.strategy = s;
        }
        if let Some(s) = str_field(&data, "provider")? {
            cfg.provider = s;
        }
        for (key, slot) in [("k", &mut cfg.k), ("fetch_k", &mut cfg.fetch_k), ("top_n", &mut cfg.top_n)] {
            if let Some(v) = data.get(key) {
                *slot = v.as_u64().filter(|&n| n >= 1).ok_or_else(|| RetrievalError::config(format!("{key} must be a positive integer")))?
                    as usize;
            }
        }
        cfg.score_threshold = match data.get("score_threshold") {
            None | Some(Value::Null) => None,
            Some(Value::Number(n)) => n.as_f64(),
            Some(_) => return Err(RetrievalError::config("score_threshold must be a number or None")),
        };
        cfg.collection_name = str_field(&data, "collection_name")?;
        if let Some(docs) = data.get("documents") {
            cfg.documents = match docs {
                Value::Null => Vec::new(),
                Value::Array(items) => items.iter().cloned().map(as_document).collect(),
                other => vec![as_document(other.clone())],
            };
        }
        if let Some(r) = data.get("retrievers") {
            cfg.retrievers = match r {
                Value::Null => Vec::new(),
                Value::Array(items) => items
                    .iter()
                    .map(|v| v.as_str().map(str::to_string).ok_or_else(|| RetrievalError::config("retrievers must be strategy names (inject objects with with_component)")))
                    .collect::<Result<_>>()?,
                _ => return Err(RetrievalError::config("retrievers must be a list")),
            };
        }
        if let Some(w) = data.get("weights") {
            cfg.weights = match w {
                Value::Null => Vec::new(),
                Value::Array(items) => items.iter().map(|v| v.as_f64().ok_or_else(|| RetrievalError::config("weights must be numbers"))).collect::<Result<_>>()?,
                _ => return Err(RetrievalError::config("weights must be a list")),
            };
            if cfg.weights.iter().any(|&x| x < 0.0) {
                return Err(RetrievalError::config("weights must be non-negative"));
            }
        }
        cfg.filters = match data.get("filters") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => return Err(RetrievalError::config("filters must be a mapping")),
        };
        if let Some(s) = str_field(&data, "rerank_model")? {
            cfg.rerank_model = s;
        }
        for key in ["embedding", "llm"] {
            match data.get(key) {
                None | Some(Value::Null) => {}
                Some(Value::Object(m)) => {
                    let v = Some(Value::Object(m.clone()));
                    if key == "embedding" { cfg.embedding = v } else { cfg.llm = v }
                }
                Some(Value::String(s)) if key == "embedding" => cfg.embedding = Some(Value::String(s.clone())),
                Some(_) => return Err(RetrievalError::config(format!("{key} must be a configuration mapping (inject objects with with_{key})"))),
            }
        }
        cfg.provider_options = match data.get("provider_options") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(_) => return Err(RetrievalError::config("provider_options must be a mapping")),
        };
        cfg.strategy = normalize_strategy(&cfg.strategy)?.to_string();
        cfg.provider = normalize_provider(&cfg.provider)?.to_string();
        Ok(cfg)
    }

    /// Cópia com alterações só desta chamada (`retrieve(query, strategy=..., k=...)` do Python).
    /// Chaves legadas nas alterações valem pelo nome atual (no Python, `k_neighbors` numa
    /// chamada caía em `options`, porque `k` já existia). O acervo não é reserializado.
    pub fn with_overrides(&self, overrides: Value) -> Result<Self> {
        let Value::Object(mut updates) = overrides else {
            return Err(RetrievalError::config("overrides must be a mapping"));
        };
        for (old, new) in LEGACY {
            if !updates.contains_key(*new)
                && let Some(v) = updates.remove(*old)
            {
                updates.insert(new.to_string(), v);
            }
        }
        let mut base = self.clone();
        let documents = std::mem::take(&mut base.documents);
        let mut data = match serde_json::to_value(&base).map_err(|e| RetrievalError::config(e.to_string()))? {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        data.remove("documents");
        let replaces_documents = updates.contains_key("documents");
        data.extend(updates);
        let mut cfg = Self::from_value(Value::Object(data))?;
        if !replaces_documents {
            cfg.documents = documents;
        }
        Ok(cfg)
    }

    pub fn option(&self, key: &str) -> Option<&Value> {
        self.options.get(key).filter(|v| !v.is_null())
    }

    pub fn option_f64(&self, key: &str, default: f64) -> f64 {
        self.option(key).and_then(Value::as_f64).unwrap_or(default)
    }

    pub fn option_str(&self, key: &str) -> Option<&str> {
        self.option(key).and_then(Value::as_str)
    }

    /// Lista de nomes em `options[key]`.
    pub fn option_strings(&self, key: &str) -> Vec<String> {
        match self.option(key) {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn aliases_and_legacy_parameters() {
        let c = RetrievalConfig::from_value(json!({
            "retrieval_strategy": "reciprocal-rank-fusion",
            "retriever_provider": "Memory",
            "k_neighbors": 7,
            "k_neighbors_rerank": 30,
            "top_n_rerank": 4,
            "rrf_k": 10,
        }))
        .unwrap();
        assert_eq!((c.strategy.as_str(), c.provider.as_str(), c.k, c.fetch_k, c.top_n), ("rrf", "in_memory", 7, 30, 4));
        assert_eq!(c.options["rrf_k"], 10);
        assert_eq!(RetrievalConfig::new("mmr").unwrap().strategy, "dense_mmr");
    }

    #[test]
    fn validation() {
        for bad in [json!({"strategy": "unknown"}), json!({"k": 0}), json!({"fetch_k": true}), json!({"score_threshold": "x"}), json!({"weights": [-1]})] {
            assert!(RetrievalConfig::from_value(bad.clone()).is_err(), "{bad}");
        }
        assert!(matches!(RetrievalConfig::from_value(json!({"provider": "pinecone"})), Err(RetrievalError::ExternalProvider(_))));
        assert!(matches!(RetrievalConfig::from_value(json!({"provider": "postgres"})), Err(RetrievalError::ExternalProvider(_))));
        assert!(matches!(RetrievalConfig::from_value(json!({"provider": "nope"})), Err(RetrievalError::UnsupportedProvider(..))));
        assert_eq!(SUPPORTED_STRATEGIES.len(), 44);
        assert_eq!(SUPPORTED_PROVIDERS.len() + EXTERNAL_PROVIDERS.len(), 118);
    }

    #[test]
    fn overrides_do_not_mutate() {
        let c = RetrievalConfig::from_value(json!({"strategy": "dense", "k": 1, "options": {"rrf_k": 5}})).unwrap();
        let o = c.with_overrides(json!({"strategy": "bm25", "k": 2, "b": 0.5})).unwrap();
        assert_eq!((o.strategy.as_str(), o.k), ("bm25", 2));
        assert_eq!((o.options["rrf_k"].as_i64(), o.options["b"].as_f64()), (Some(5), Some(0.5)));
        assert_eq!((c.strategy.as_str(), c.k), ("dense", 1));
    }
}
