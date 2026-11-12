//! Recuperação (retrieval): `GET /v1/retrieval/strategies` (catálogo) e `POST /v1/retrieval`
//! (os documentos mais relevantes de um acervo para cada consulta).
//!
//! A API não guarda índice: o acervo vem na requisição (ex.: os `chunks` de `POST /v1/chunk`).
//! Como no embedding, a busca roda num processo worker isolado (`worker.rs`): sem segredos, sem
//! rede, com prazo e teto de memória. Sem rede, nada é baixado nem nenhum LLM é chamado:
//! embeddings usam os modelos LOCAIS de `EMBEDDING_MODELS` (ou estratégias sem modelo), o
//! reranking usa `RETRIEVAL_RERANK_MODEL`, e as estratégias que reescrevem a consulta recebem o
//! texto já gerado pelo LLM do cliente (`generated_queries`). Fontes externas (grafo, SQL, web),
//! roteadores e retrievers injetados só existem na biblioteca.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use retriever::{Document, QueryGenerator, RetrievalError, RetrievalManager, Retriever};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::auth::Auth;
use crate::chunk::read_json;
use crate::config::{EmbeddingSettings, RetrievalSettings};
use crate::error::ApiError;
use crate::models::{
    ErrorBody, RetrievalLimits, RetrievalRequest, RetrievalResponse, RetrievalResult, RetrievalStrategiesResponse, RetrievalStrategyInfo,
    RetrievedDocument,
};
use crate::state::AppState;
use crate::worker::{self, JobKind, WorkerFailure};

/// Itens de cada lista de nomes (`components`, `strategies`, `stages`) e de `weights`.
const MAX_LIST: usize = 8;
/// Vocabulário/hash padrão e máximo das estratégias de embedding esparsas usadas como densas.
const SPARSE_DIMENSIONS: usize = 4096;
const MAX_SPARSE_DIMENSIONS: usize = 16_384;

const E: &str = "embedding";
const G: &str = "generated_queries";
const R: &str = "rerank_model";

const LIBRARY_ONLY: &str = "fonte ou objeto injetado: só pela biblioteca retriever";
const NEEDS_LLM: &str = "precisa de um LLM, e o worker não tem rede";

struct Spec {
    name: &'static str,
    family: &'static str,
    requires: &'static [&'static str],
    /// `Some(motivo)`: nunca disponível pela API.
    unsupported: Option<&'static str>,
    description: &'static str,
}

const fn spec(name: &'static str, family: &'static str, requires: &'static [&'static str], description: &'static str) -> Spec {
    Spec { name, family, requires, unsupported: None, description }
}

const fn library(name: &'static str, family: &'static str, reason: &'static str, description: &'static str) -> Spec {
    Spec { name, family, requires: &[], unsupported: Some(reason), description }
}

/// Uma linha por estratégia do crate `retriever`, na mesma ordem (um teste confere).
const SPECS: &[Spec] = &[
    spec("dense", "semantic", &[E], "Cosseno entre o embedding da consulta e o de cada documento; `score_threshold` opcional."),
    spec("dense_mmr", "semantic", &[E], "MMR: relevância com diversidade (`options.lambda_mult`, 0,5)."),
    spec("similarity_threshold", "semantic", &[E], "Só documentos com nota de pelo menos `score_threshold`."),
    library(
        "multimodal",
        "semantic",
        "imagens por caminho de arquivo não são aceitas pela API; use POST /v1/embedding com clip/siglip",
        "Texto buscando imagens (CLIP/SigLIP).",
    ),
    spec("sparse", "lexical", &[], "Busca esparsa: BM25 sobre o acervo."),
    spec("bm25", "lexical", &[], "Okapi BM25 (`options.k1` 1,5, `options.b` 0,75)."),
    spec("bm25_plus", "lexical", &[], "BM25+ (`options.delta` 1): não penaliza demais documentos longos."),
    spec("tfidf", "lexical", &[], "TF-IDF do scikit-learn (palavras e pares de palavras)."),
    spec("keyword", "lexical", &[], "Contagem dos termos da consulta no texto."),
    spec("boolean", "lexical", &[], "Todos os termos (`options.operator` = `and`) ou algum (`or`), sem nota."),
    spec("fuzzy", "lexical", &[], "Semelhança aproximada (difflib): tolera erros de digitação em textos curtos."),
    spec("metadata", "lexical", &[], "Só `filters`: ignora o texto da consulta."),
    spec("hybrid", "fusion", &[E], "Reciprocal Rank Fusion de `components` (padrão `dense` + `bm25`)."),
    spec("weighted_hybrid", "fusion", &[E], "RRF com `weights` por componente."),
    spec("rrf", "fusion", &[E], "Reciprocal Rank Fusion com `weights` (`options.rrf_k`, 60)."),
    spec("ensemble", "fusion", &[E], "EnsembleRetriever: RRF dos componentes com `weights`."),
    spec("multi_query", "query", &[G, E], "Busca a consulta e as variações de `generated_queries` e funde por RRF."),
    spec("query_expansion", "query", &[G, E], "Como `multi_query`, com consultas de sinônimos e termos relacionados."),
    spec("rephrase_query", "query", &[G, E], "Busca a reescrita da consulta (primeira de `generated_queries`)."),
    spec("hyde", "query", &[G, E], "Busca um documento hipotético que responderia a pergunta (primeira de `generated_queries`)."),
    spec("step_back", "query", &[G, E], "Busca uma pergunta mais geral (primeira de `generated_queries`)."),
    spec("decomposition", "query", &[G, E], "Busca cada subpergunta de `generated_queries` e funde por RRF."),
    library("self_query", "query", NEEDS_LLM, "Consulta em linguagem natural vira texto de busca e filtro de metadados (envie `filters`)."),
    spec(
        "parent_document",
        "hierarchical",
        &[E],
        "Acha o trecho e devolve o pai: `options.parent_store` ou `metadata.parent_content` (chunking `parent_child`).",
    ),
    spec("multi_vector", "hierarchical", &[E], "Várias representações por documento (`parent_id`) devolvem o original."),
    spec("small_to_big", "hierarchical", &[E], "Trechos pequenos para buscar, o pai para responder."),
    library("contextual_compression", "rerank", NEEDS_LLM, "Reduz cada candidato ao trecho relevante para a consulta."),
    spec("rerank", "rerank", &[R, E], "Reordena `fetch_k` candidatos com o cross-encoder e devolve `top_n`."),
    spec("cross_encoder_rerank", "rerank", &[R, E], "Reordena com o cross-encoder (consulta e documento juntos no modelo)."),
    library("colbert_rerank", "rerank", "ColBERT não está incluído; só pela biblioteca, com um reranker injetado", "Reordena com ColBERT."),
    library("llm_rerank", "rerank", NEEDS_LLM, "O LLM dá uma nota de 0 a 10 a cada candidato."),
    spec("diversity", "rerank", &[E], "MMR sobre a busca semântica: sem trechos quase iguais."),
    spec("time_weighted", "time", &[E], "Relevância somada à recência dos candidatos."),
    spec("recency", "time", &[], "Os mais recentes: `exp(-decay_rate * dias)` pela data em `metadata.timestamp` (`options.time_key`)."),
    library("graph", "source", LIBRARY_ONLY, "Busca num grafo (GraphRAG)."),
    library("knowledge_graph", "source", LIBRARY_ONLY, "Busca num grafo de conhecimento."),
    library("sql", "source", LIBRARY_ONLY, "Text-to-SQL sobre um banco."),
    library("web_search", "source", LIBRARY_ONLY, "Busca na web."),
    library("knowledge_base", "source", LIBRARY_ONLY, "Base de conhecimento gerenciada."),
    spec("adaptive", "orchestration", &[], "Com `embedding`: consultas com números ou códigos vão para `hybrid`, as outras para `dense`; sem, `bm25`."),
    library("router", "orchestration", LIBRARY_ONLY, "Um roteador escolhe a estratégia de cada consulta."),
    spec("fallback", "orchestration", &[], "A primeira de `options.strategies` (padrão `dense`, `bm25`) que devolver algo."),
    spec("cascade", "orchestration", &[E, R], "Estágios (`options.stages`, padrão `dense`, `rerank`): busca, depois reordena, comprime ou funde."),
    library("custom", "orchestration", LIBRARY_ONLY, "Retriever próprio."),
];

/// Catálogo com a disponibilidade DESTE servidor.
pub struct Catalog {
    response: RetrievalStrategiesResponse,
}

impl Catalog {
    pub fn new(settings: &RetrievalSettings, embedding: &crate::embed::Catalog) -> Self {
        let strategies: Vec<RetrievalStrategyInfo> = SPECS
            .iter()
            .map(|s| {
                let rerank_missing = settings.rerank_model.is_none() && matches!(s.name, "rerank" | "cross_encoder_rerank");
                let unavailable = s.unsupported.or(rerank_missing.then_some("o servidor não tem modelo de reranking local (RETRIEVAL_RERANK_MODEL)"));
                RetrievalStrategyInfo {
                    name: s.name.to_string(),
                    aliases: retriever::config::STRATEGY_ALIASES.iter().filter(|(_, c)| *c == s.name).map(|(a, _)| a.to_string()).collect(),
                    family: s.family.to_string(),
                    description: s.description.to_string(),
                    available: unavailable.is_none(),
                    unavailable_reason: unavailable.map(str::to_string),
                    requires: s.requires.iter().map(|r| r.to_string()).collect(),
                }
            })
            .collect();
        Self {
            response: RetrievalStrategiesResponse {
                count: strategies.len(),
                strategies,
                limits: RetrievalLimits {
                    max_documents: settings.max_documents,
                    max_text_bytes: settings.max_text_bytes,
                    max_queries: settings.max_queries,
                    max_query_bytes: settings.max_query_bytes,
                    max_k: settings.max_k,
                    max_fetch_k: settings.max_fetch_k,
                },
                embedding_strategies: embedding.text_strategies(),
                rerank_model: settings.rerank_model.as_deref().map(crate::embed::display_model),
            },
        }
    }

    fn get(&self, name: &str) -> Option<&RetrievalStrategyInfo> {
        self.response.strategies.iter().find(|s| s.name == name)
    }
}

// ------------------------------------------------------------------ protocolo com o worker

#[derive(Debug, Serialize, Deserialize)]
pub struct Job {
    /// Configuração do `RetrievalManager`, com o acervo e os caminhos dos modelos.
    pub config: Value,
    pub queries: Vec<String>,
    pub generated_queries: Vec<String>,
}

/// Motivo de recusa (422). O worker devolve só o código; a mensagem é desta tabela fixa.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    /// A estratégia (ou um componente, estágio ou base) precisa de `embedding`.
    NeedsEmbedding,
    InvalidOptions,
    /// Nenhuma estratégia do `fallback` devolveu resultado.
    NothingFound,
    Failed,
}

impl Rejection {
    pub fn message(self) -> &'static str {
        match self {
            Rejection::NeedsEmbedding => {
                "esta estratégia (ou um componente, estágio ou estratégia base) usa busca densa: envie `embedding` (veja GET /v1/retrieval/strategies)"
            }
            Rejection::InvalidOptions => "opções inválidas para esta estratégia (componentes, pesos, estágios ou estratégia base)",
            Rejection::NothingFound => "nenhuma estratégia de `options.strategies` devolveu resultado com estes parâmetros",
            Rejection::Failed => "não foi possível buscar com essa estratégia e esses parâmetros",
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Timings {
    pub confine_ms: u64,
    /// Carga dos modelos (embedding e cross-encoder).
    pub load_ms: u64,
    pub retrieve_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Reply {
    Ok {
        results: Vec<Vec<Document>>,
        #[serde(default)]
        timings: Timings,
    },
    Rejected {
        reason: Rejection,
    },
}

fn ms(d: Duration) -> u64 {
    d.as_millis().try_into().unwrap_or(u64::MAX)
}

/// Consultas geradas pelo cliente no lugar do LLM.
struct Generated(Vec<String>);

impl QueryGenerator for Generated {
    fn generate(&self, _query: &str, _mode: &str, count: Option<usize>) -> retriever::Result<Vec<String>> {
        Ok(self.0.iter().take(count.unwrap_or(1)).cloned().collect())
    }
}

fn rejected(stage: &str, e: RetrievalError) -> Result<Reply, String> {
    // Vai para o stderr do worker (log da API); nunca para o cliente.
    eprintln!("{stage}: {e}");
    let reason = match e {
        RetrievalError::MissingDependency(_) => Rejection::NeedsEmbedding,
        RetrievalError::InvalidConfig(_) | RetrievalError::UnsupportedStrategy(..) | RetrievalError::Input(_) => Rejection::InvalidOptions,
        RetrievalError::AllFailed(_) => Rejection::NothingFound,
        _ => Rejection::Failed,
    };
    Ok(Reply::Rejected { reason })
}

/// Executa o job DENTRO do worker (já confinado). `Err` = falha interna.
pub fn execute_job(path: &Path) -> Result<Reply, String> {
    let raw = std::fs::read(path).map_err(|e| format!("leitura do job: {e}"))?;
    let job: Job = serde_json::from_slice(&raw).map_err(|e| format!("job inválido: {e}"))?;
    drop(raw);

    let mut timings = Timings::default();
    let t = Instant::now();
    let mut manager = match RetrievalManager::from_value(job.config) {
        Ok(m) => m,
        Err(e) => return rejected("configuração", e),
    };
    if !job.generated_queries.is_empty() {
        manager = manager.with_query_generator(Arc::new(Generated(job.generated_queries)));
    }
    if let Err(e) = manager.load_retriever() {
        return rejected("carga dos modelos", e);
    }
    timings.load_ms = ms(t.elapsed());

    let t = Instant::now();
    let mut results = Vec::with_capacity(job.queries.len());
    for query in &job.queries {
        match manager.retrieve(query) {
            Ok(docs) => results.push(docs),
            Err(e) => return rejected(&format!("busca {}", manager.config().strategy), e),
        }
    }
    timings.retrieve_ms = ms(t.elapsed());
    Ok(Reply::Ok { results, timings })
}

// ------------------------------------------------------------------ validação

/// Opções aceitas do cliente (as demais, como `device`, são do servidor).
const BAD_OPTIONS: &str = "options inválidas; aceitas: lambda_mult, k1, b, delta, operator, rrf_k, query_count, base_strategy, parent_id_key, parent_store, time_key, decay_rate, strategies, stages";

fn number_in(v: &Value, min: f64, max: f64) -> bool {
    v.as_f64().is_some_and(|x| x.is_finite() && (min..=max).contains(&x))
}

fn short_str(v: &Value) -> bool {
    v.as_str().is_some_and(|s| !s.is_empty() && s.len() <= 64)
}

/// Estratégia usada dentro de outra (`components`, `strategies`, `stages`, `base_strategy`):
/// precisa existir e estar disponível pela API.
fn nested(name: &Value, catalog: &Catalog, has_generated: bool) -> Result<String, ApiError> {
    const MSG: &str = "estratégia indisponível em components, options.strategies, options.stages ou options.base_strategy; veja GET /v1/retrieval/strategies";
    let raw = name.as_str().ok_or(ApiError::Unprocessable(MSG))?;
    let canonical = retriever::normalize_strategy(raw).map_err(|_| ApiError::Unprocessable(MSG))?;
    let info = catalog.get(canonical).filter(|i| i.available).ok_or(ApiError::Unprocessable(MSG))?;
    if info.requires.iter().any(|r| r == G) && !has_generated {
        return Err(ApiError::Unprocessable("uma estratégia de options usa `generated_queries`: envie as consultas geradas"));
    }
    Ok(canonical.to_string())
}

fn names(v: &Value, catalog: &Catalog, has_generated: bool) -> Result<Vec<String>, ApiError> {
    let items = v.as_array().filter(|a| !a.is_empty() && a.len() <= MAX_LIST).ok_or(ApiError::Unprocessable(BAD_OPTIONS))?;
    items.iter().map(|n| nested(n, catalog, has_generated)).collect()
}

/// Valida `options` e devolve a versão normalizada (nomes de estratégia canônicos).
fn check_options(mut options: Map<String, Value>, settings: &RetrievalSettings, catalog: &Catalog, has_generated: bool) -> Result<Map<String, Value>, ApiError> {
    for (key, v) in options.iter_mut() {
        let ok = match key.as_str() {
            "lambda_mult" | "b" => number_in(v, 0.0, 1.0),
            "k1" | "delta" => number_in(v, 0.0, 10.0),
            "rrf_k" => number_in(v, 1.0, 10_000.0),
            "decay_rate" => number_in(v, 0.0, 1_000.0),
            "operator" => matches!(v.as_str(), Some("and" | "or")),
            "query_count" => v.as_u64().is_some_and(|n| n >= 1 && n as usize <= settings.max_queries),
            "parent_id_key" | "time_key" => short_str(v),
            "parent_store" => v.is_object(),
            "base_strategy" => {
                *v = Value::String(nested(v, catalog, has_generated)?);
                true
            }
            "strategies" | "stages" => {
                *v = json!(names(v, catalog, has_generated)?);
                true
            }
            _ => false,
        };
        if !ok {
            return Err(ApiError::Unprocessable(BAD_OPTIONS));
        }
    }
    Ok(options)
}

/// `embedding` da requisição → configuração do crate `embedding` (com o modelo local) e nome
/// para a resposta.
fn embedding_config(
    req: crate::models::RetrievalEmbedding,
    settings: &EmbeddingSettings,
    catalog: &crate::embed::Catalog,
) -> Result<(Value, String), ApiError> {
    let name = embedding::normalize_strategy(&req.strategy)
        .map_err(|_| ApiError::Unprocessable("estratégia de embedding desconhecida ou paga; veja GET /v1/embedding/strategies"))?;
    let info = catalog
        .get(name)
        .filter(|i| i.available)
        .ok_or(ApiError::Unprocessable("estratégia de embedding indisponível neste servidor; veja GET /v1/retrieval/strategies"))?;
    if name == "sparse" {
        return Err(ApiError::Unprocessable("a estratégia de embedding `sparse` (SPLADE) não serve para a busca densa; use bm25, hashing ou tfidf"));
    }
    let (dimensions, max) = if info.kind == "sparse" {
        (Some(req.dimensions.unwrap_or(SPARSE_DIMENSIONS)), MAX_SPARSE_DIMENSIONS)
    } else {
        (req.dimensions, crate::embed::MAX_DIMENSIONS)
    };
    if dimensions.is_some_and(|d| d == 0 || d > max) {
        return Err(ApiError::Unprocessable("embedding.dimensions fora do intervalo"));
    }
    let mut cfg = json!({
        "strategy": name,
        "device": settings.device,
        "batch_size": settings.batch_size,
        "normalize": req.normalize.unwrap_or(false),
        "dimensions": dimensions,
    });
    if let Some(dir) = settings.models.get(name) {
        cfg["model_name"] = json!(dir.to_string_lossy());
    }
    Ok((cfg, info.model.clone().unwrap_or_else(|| name.to_string())))
}

/// O que sai da validação: o job e os caminhos que o worker precisa ler.
pub struct Prepared {
    pub job: Job,
    pub strategy: String,
    pub embedding: Option<String>,
    pub model_dir: Option<PathBuf>,
}

/// Valida o pedido contra o catálogo e os limites e monta o job.
fn prepare(
    req: RetrievalRequest,
    settings: &RetrievalSettings,
    embedding_settings: &EmbeddingSettings,
    catalog: &Catalog,
    embedding_catalog: &crate::embed::Catalog,
) -> Result<Prepared, ApiError> {
    let name = retriever::normalize_strategy(&req.strategy)
        .map_err(|_| ApiError::Unprocessable("estratégia desconhecida; veja GET /v1/retrieval/strategies"))?;
    let info = catalog.get(name).ok_or(ApiError::Unprocessable("estratégia desconhecida; veja GET /v1/retrieval/strategies"))?;
    if !info.available {
        return Err(ApiError::Unprocessable("estratégia indisponível neste servidor; veja GET /v1/retrieval/strategies"));
    }

    // Consultas.
    let queries = match (req.query, req.queries) {
        (Some(q), None) => vec![q],
        (None, Some(q)) => q,
        _ => return Err(ApiError::BadRequest("envie `query` ou `queries` (um deles)")),
    };
    if queries.is_empty() {
        return Err(ApiError::Unprocessable("a lista de consultas está vazia"));
    }
    if queries.len() > settings.max_queries {
        return Err(ApiError::Unprocessable("consultas demais numa requisição; veja GET /v1/retrieval/strategies"));
    }
    let generated = req.generated_queries.unwrap_or_default();
    for q in queries.iter().chain(&generated) {
        if q.trim().is_empty() {
            return Err(ApiError::Unprocessable("consulta vazia"));
        }
        if q.len() > settings.max_query_bytes {
            return Err(ApiError::PayloadTooLarge);
        }
    }
    let wants_generated = info.requires.iter().any(|r| r == G);
    if wants_generated && generated.is_empty() {
        return Err(ApiError::Unprocessable(
            "esta estratégia usa consultas geradas por LLM: envie-as em `generated_queries` (o servidor não chama LLMs)",
        ));
    }
    if !generated.is_empty() && queries.len() > 1 {
        return Err(ApiError::Unprocessable("`generated_queries` vale para uma única `query`"));
    }
    if generated.len() > settings.max_queries {
        return Err(ApiError::Unprocessable("consultas demais numa requisição; veja GET /v1/retrieval/strategies"));
    }

    // Acervo.
    if req.documents.is_empty() {
        return Err(ApiError::Unprocessable("a lista de documentos está vazia"));
    }
    if req.documents.len() > settings.max_documents {
        return Err(ApiError::Unprocessable("documentos demais numa requisição; veja GET /v1/retrieval/strategies"));
    }
    let options = req.options.unwrap_or_default();
    let mut bytes = serde_json::to_vec(&options).map_or(usize::MAX, |v| v.len());
    for d in &req.documents {
        bytes = bytes.saturating_add(d.page_content.len()).saturating_add(serde_json::to_vec(&d.metadata).map_or(usize::MAX, |v| v.len()));
    }
    if bytes > settings.max_text_bytes {
        return Err(ApiError::PayloadTooLarge);
    }

    // Parâmetros.
    let in_range = |v: Option<usize>, max: usize| v.is_none_or(|n| (1..=max).contains(&n));
    if !in_range(req.k, settings.max_k) || !in_range(req.top_n, settings.max_k) || !in_range(req.fetch_k, settings.max_fetch_k) {
        return Err(ApiError::Unprocessable("k, fetch_k ou top_n fora do intervalo; veja GET /v1/retrieval/strategies"));
    }
    if req.weights.as_ref().is_some_and(|w| w.len() > MAX_LIST || w.iter().any(|x| !x.is_finite() || *x < 0.0)) {
        return Err(ApiError::Unprocessable("weights precisa de até 8 números não negativos"));
    }
    let mut options = check_options(options, settings, catalog, !generated.is_empty())?;
    if let Some(c) = &req.components {
        let components = names(&json!(c), catalog, !generated.is_empty())?;
        if components.iter().any(|n| retriever::FUSION_STRATEGIES.contains(&n.as_str())) {
            return Err(ApiError::Unprocessable("components não podem ser estratégias de fusão (hybrid, weighted_hybrid, rrf, ensemble)"));
        }
        options.insert("components".into(), json!(components));
    }
    if name == "cascade" && !options.contains_key("stages") && settings.rerank_model.is_none() {
        return Err(ApiError::Unprocessable("cascade usa `rerank` por padrão e o servidor não tem modelo de reranking: envie options.stages"));
    }
    if !generated.is_empty() && !options.contains_key("query_count") {
        options.insert("query_count".into(), json!(generated.len()));
    }
    // Do servidor, nunca do cliente.
    options.insert("device".into(), json!(embedding_settings.device));

    let (embedding, embedding_name) = match req.embedding {
        Some(e) => {
            let (cfg, display) = embedding_config(e, embedding_settings, embedding_catalog)?;
            (Some(cfg), Some(display))
        }
        None => (None, None),
    };
    let model_dir = embedding.as_ref().and_then(|e| e["model_name"].as_str()).map(PathBuf::from);

    let mut config = json!({
        "strategy": name,
        "provider": "local",
        "k": req.k,
        "fetch_k": req.fetch_k,
        "top_n": req.top_n,
        "score_threshold": req.score_threshold,
        "filters": req.filters,
        "weights": req.weights,
        "embedding": embedding,
        "options": options,
    });
    if let Some(dir) = &settings.rerank_model {
        config["rerank_model"] = json!(dir.to_string_lossy());
    }
    // Remove os nulos (o crate usa os padrões) e confere a configuração ANTES do worker.
    if let Value::Object(m) = &mut config {
        m.retain(|_, v| !v.is_null());
    }
    retriever::RetrievalConfig::from_value(config.clone())
        .map_err(|_| ApiError::Unprocessable("parâmetros inválidos (filters, weights ou options); veja GET /v1/retrieval/strategies"))?;
    config["documents"] = Value::Array(req.documents.into_iter().map(|d| json!({"page_content": d.page_content, "metadata": d.metadata})).collect());

    Ok(Prepared {
        job: Job { config, queries, generated_queries: generated },
        strategy: name.to_string(),
        embedding: embedding_name,
        model_dir,
    })
}

/// Teto do corpo: acervo, `options` e consultas (com o escape do JSON).
fn body_limit(s: &RetrievalSettings) -> usize {
    let queries = s.max_query_bytes.saturating_mul(s.max_queries).saturating_mul(2);
    s.max_text_bytes.saturating_mul(2).saturating_add(queries.saturating_mul(2)).saturating_add(1024 * 1024)
}

// ------------------------------------------------------------------ rotas

/// Lista as estratégias de recuperação e se cada uma está disponível neste servidor.
///
/// Para cada estratégia: apelidos, família, o que ela usa com os parâmetros padrão
/// (`embedding`, `generated_queries`, `rerank_model`) e, se indisponível, o motivo. Traz também
/// as estratégias aceitas em `embedding`, o modelo de reranking e os limites.
#[utoipa::path(
    get, path = "/v1/retrieval/strategies", tag = "Recuperação",
    security(("bearer" = []), ("oauth2" = ["ocr"])),
    responses(
        (status = 200, description = "Estratégias, disponibilidade e limites", body = RetrievalStrategiesResponse),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 403, description = "Falta o escopo `ocr`", body = ErrorBody),
    )
)]
pub async fn strategies(State(st): State<AppState>, auth: Auth) -> Result<Json<RetrievalStrategiesResponse>, ApiError> {
    auth.require_scope("ocr")?;
    Ok(Json(st.retrieval_catalog.response.clone()))
}

/// Busca os documentos mais relevantes de um acervo.
///
/// Envie o acervo em `documents` (ex.: a lista `chunks` de `POST /v1/chunk`) e `query` (ou até
/// `max_queries` consultas em `queries`). Cada documento devolvido traz em `metadata` a nota
/// (`retrieval_score`, quando a estratégia dá nota), a posição (`retrieval_rank`) e a estratégia
/// que o encontrou (`retrieval_strategy`).
///
/// Estratégias densas, fusões e as que refinam candidatos (`options.base_strategy`, padrão
/// `dense`) precisam de `embedding`: uma estratégia de `GET /v1/embedding/strategies`
/// disponível neste servidor (`hashing`, `tfidf`, `bm25` e `lsa` não usam modelo). As que
/// reescrevem a consulta recebem o texto já gerado pelo seu LLM em `generated_queries`.
#[utoipa::path(
    post, path = "/v1/retrieval", tag = "Recuperação",
    security(("bearer" = []), ("oauth2" = ["ocr"])),
    request_body = RetrievalRequest,
    responses(
        (status = 200, description = "Um resultado por consulta, na ordem enviada", body = RetrievalResponse),
        (status = 400, description = "JSON inválido, campo desconhecido, ou nenhuma/as duas entre `query` e `queries`", body = ErrorBody),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 403, description = "Falta o escopo `ocr`", body = ErrorBody),
        (status = 413, description = "Acervo, options ou consulta acima do limite", body = ErrorBody),
        (status = 422, description = "Estratégia desconhecida ou indisponível, parâmetro inválido, falta `embedding` ou `generated_queries`, ou falha da busca", body = ErrorBody),
        (status = 429, description = "Limite de taxa ou busca já em andamento para este usuário", body = ErrorBody),
        (status = 503, description = "Servidor ocupado, sem memória ou tempo esgotado", body = ErrorBody),
    )
)]
pub async fn retrieve(State(st): State<AppState>, auth: Auth, headers: HeaderMap, body: Body) -> Result<Json<RetrievalResponse>, ApiError> {
    let started = Instant::now();
    auth.require_scope("ocr")?;
    let user = auth.user_id()?;

    // 1) Limites ANTES de ler o corpo.
    st.limiters.check_retrieval(user).await.map_err(ApiError::TooManyRequests)?;
    let guard = st.retrieval.try_user_guard(user).ok_or(ApiError::TooManyRequests(Duration::from_secs(2)))?;

    // 2) Corpo e validação (barata, na API).
    let cfg = &st.cfg.retrieval;
    let req: RetrievalRequest = read_json(&headers, body, body_limit(cfg)).await?;
    let prepared = prepare(req, cfg, &st.cfg.embedding, &st.retrieval_catalog, &st.embedding_catalog)?;
    let Prepared { job, strategy, embedding, model_dir } = prepared;
    let queries = job.queries.clone();
    let documents = job.config["documents"].as_array().map_or(0, Vec::len);

    // 3) Vaga global.
    if st.memory.is_under_pressure() {
        tracing::warn!("busca recusada: memória acima do limite configurado");
        return Err(ApiError::Unavailable(Duration::from_secs(15)));
    }
    let permit = match tokio::time::timeout(Duration::from_secs(2), st.retrieval.slots.clone().acquire_owned()).await {
        Ok(Ok(p)) => p,
        _ => return Err(ApiError::Unavailable(Duration::from_secs(5))),
    };

    // 4) Worker isolado; o job vai num arquivo do diretório exclusivo, apagado no fim.
    let tmp = tempfile::Builder::new().prefix("retrieve-").tempdir().map_err(|e| ApiError::internal("criar diretório temporário", e))?;
    let path = tmp.path().join("job.json");
    let data = serde_json::to_vec(&job).map_err(|e| ApiError::internal("serializar job", e))?;
    drop(job);
    tokio::fs::write(&path, data).await.map_err(|e| ApiError::internal("gravar job", e))?;
    let t_worker = Instant::now();
    let outcome = worker::run_retrieval_job(&st.cfg, tmp.path(), &path, model_dir.as_deref()).await;
    let worker_ms = ms(t_worker.elapsed());
    drop((permit, guard, tmp));

    let results = match outcome {
        Ok(Reply::Ok { results, timings: w }) => {
            tracing::info!(
                strategy = %strategy,
                documents,
                queries = queries.len(),
                worker_ms,
                worker_spawn_ms = worker_ms.saturating_sub(w.confine_ms + w.load_ms + w.retrieve_ms),
                confine_ms = w.confine_ms,
                load_ms = w.load_ms,
                retrieve_ms = w.retrieve_ms,
                total_ms = ms(started.elapsed()),
                "busca concluída"
            );
            results
        }
        Ok(Reply::Rejected { reason }) => {
            if reason == Rejection::Failed {
                tracing::warn!(strategy = %strategy, "busca falhou");
            }
            return Err(ApiError::Unprocessable(reason.message()));
        }
        Err(WorkerFailure::Timeout) => return Err(ApiError::Unavailable(Duration::from_secs(10))),
        Err(WorkerFailure::ReplyTooLarge) => {
            return Err(ApiError::Unprocessable("resposta grande demais; reduza k ou o número de consultas"));
        }
        Err(failure) => {
            crate::security::event(&st, failure.event_kind(JobKind::Retrieval), Some(user), None, None, Some(&failure.to_string())).await;
            return Err(ApiError::Unprocessable(Rejection::Failed.message()));
        }
    };
    if results.len() != queries.len() {
        return Err(ApiError::internal("resposta do worker de busca", "número de resultados diferente do de consultas"));
    }

    let results: Vec<RetrievalResult> = queries
        .into_iter()
        .zip(results)
        .map(|(query, docs)| RetrievalResult {
            query,
            count: docs.len(),
            documents: docs.into_iter().map(|d| RetrievedDocument { page_content: d.page_content, metadata: d.metadata }).collect(),
        })
        .collect();
    Ok(Json(RetrievalResponse { strategy, embedding, count: results.len(), results }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{InputDocument, RetrievalEmbedding};

    fn catalogs(s: &RetrievalSettings) -> (Catalog, crate::embed::Catalog) {
        let emb = crate::embed::Catalog::new(&EmbeddingSettings::default());
        (Catalog::new(s, &emb), emb)
    }

    fn req(strategy: &str) -> RetrievalRequest {
        RetrievalRequest {
            strategy: strategy.into(),
            query: Some("praias".into()),
            queries: None,
            documents: vec![InputDocument { page_content: "Fortaleza tem praias.".into(), metadata: Map::new() }],
            k: None,
            fetch_k: None,
            top_n: None,
            score_threshold: None,
            filters: None,
            components: None,
            weights: None,
            embedding: None,
            generated_queries: None,
            options: None,
        }
    }

    fn run(r: RetrievalRequest, s: &RetrievalSettings) -> Result<Prepared, ApiError> {
        let (c, e) = catalogs(s);
        prepare(r, s, &EmbeddingSettings::default(), &c, &e)
    }

    fn unprocessable(r: Result<Prepared, ApiError>) -> &'static str {
        match r {
            Err(ApiError::Unprocessable(m)) => m,
            Err(other) => panic!("esperava 422, veio {other:?}"),
            Ok(_) => panic!("esperava 422, veio Ok"),
        }
    }

    #[test]
    fn catalog_covers_every_strategy_in_order() {
        let names: Vec<&str> = SPECS.iter().map(|s| s.name).collect();
        assert_eq!(names, retriever::SUPPORTED_STRATEGIES);
        let (c, _) = catalogs(&RetrievalSettings::default());
        assert!(c.get("bm25").unwrap().available && c.get("hybrid").unwrap().available);
        for n in ["rerank", "cross_encoder_rerank", "graph", "router", "custom", "llm_rerank", "multimodal", "self_query"] {
            assert!(!c.get(n).unwrap().available, "{n}");
        }
        assert_eq!(c.response.embedding_strategies, ["tfidf", "hashing", "bm25", "lsa"]);
        assert!(c.response.rerank_model.is_none());
    }

    #[test]
    fn validation() {
        let s = RetrievalSettings { max_queries: 2, max_query_bytes: 10, max_k: 3, max_text_bytes: 200, ..Default::default() };
        let p = run(RetrievalRequest { strategy: "fusion".into(), components: Some(vec!["bm25".into(), "fts".into()]), ..req("x") }, &s).unwrap();
        assert_eq!(p.strategy, "rrf");
        assert_eq!(p.job.config["options"]["components"], json!(["bm25", "keyword"]));
        assert_eq!(p.job.config["options"]["device"], "cpu");
        assert!(unprocessable(run(req("nope"), &s)).contains("desconhecida"));
        assert!(unprocessable(run(req("graph"), &s)).contains("indisponível"));
        assert!(unprocessable(run(req("rerank"), &s)).contains("indisponível"));
        assert!(unprocessable(run(req("hyde"), &s)).contains("generated_queries"));
        let both = RetrievalRequest { queries: Some(vec!["a".into()]), ..req("bm25") };
        assert!(matches!(run(both, &s), Err(ApiError::BadRequest(_))));
        let many = RetrievalRequest { query: None, queries: Some(vec!["a".into(); 3]), ..req("bm25") };
        assert!(unprocessable(run(many, &s)).contains("demais"));
        assert!(matches!(run(RetrievalRequest { query: Some("x".repeat(11)), ..req("bm25") }, &s), Err(ApiError::PayloadTooLarge)));
        assert!(unprocessable(run(RetrievalRequest { k: Some(4), ..req("bm25") }, &s)).contains("intervalo"));
        let opts = |o: Value| RetrievalRequest { options: Some(o.as_object().unwrap().clone()), ..req("bm25") };
        assert!(unprocessable(run(opts(json!({"device": "cuda"})), &s)).contains("options"));
        assert!(unprocessable(run(opts(json!({"k1": -1})), &s)).contains("options"));
        assert!(unprocessable(run(opts(json!({"stages": ["graph"]})), &s)).contains("indisponível"));
        assert!(unprocessable(run(opts(json!({"base_strategy": "hyde"})), &s)).contains("generated_queries"));
        assert!(unprocessable(run(req("cascade"), &s)).contains("stages"));
        let generated = RetrievalRequest { generated_queries: Some(vec!["a".into(), "b".into()]), ..req("multi_query") };
        assert_eq!(run(generated, &s).unwrap().job.config["options"]["query_count"], 2);
        let big = RetrievalRequest {
            documents: vec![InputDocument { page_content: "x".repeat(201), metadata: Map::new() }],
            ..req("bm25")
        };
        assert!(matches!(run(big, &s), Err(ApiError::PayloadTooLarge)));
        let emb = |strategy: &str, dimensions: Option<usize>| RetrievalRequest {
            embedding: Some(RetrievalEmbedding { strategy: strategy.into(), dimensions, normalize: None }),
            ..req("dense")
        };
        let p = run(emb("feature_hashing", None), &s).unwrap();
        assert_eq!((p.embedding.as_deref(), p.job.config["embedding"]["dimensions"].as_u64()), (Some("hashing"), Some(4096)));
        assert!(p.model_dir.is_none());
        assert!(unprocessable(run(emb("e5", None), &s)).contains("indisponível"));
        assert!(unprocessable(run(emb("openai", None), &s)).contains("paga"));
        assert!(unprocessable(run(emb("hashing", Some(100_000)), &s)).contains("dimensions"));
        let bad_filter = RetrievalRequest { weights: Some(vec![-1.0]), ..req("rrf") };
        assert!(unprocessable(run(bad_filter, &s)).contains("weights"));
    }
}
