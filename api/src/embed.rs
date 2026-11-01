//! Embeddings: `GET /v1/embedding/strategies` (catálogo) e `POST /v1/embedding` (vetores de
//! textos, de `documents` devolvidos por `POST /v1/chunk` ou de imagens).
//!
//! Como o OCR e o chunking, o cálculo roda num processo worker isolado (`worker.rs`): sem
//! segredos, sem rede, com prazo e teto de memória. Sem rede, nada é baixado em tempo de
//! execução: os modelos são diretórios LOCAIS configurados pelo operador em `EMBEDDING_MODELS`
//! (`estrategia=/caminho,...`). Sem modelo, ficam `hashing` e as estratégias que aprendem o
//! vocabulário com os documentos da própria requisição (`tfidf`, `bm25`, `lsa`: envie o acervo e
//! as consultas juntos, em `texts`/`documents` e `queries`).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use embedding::{EmbeddingConfig, EmbeddingError, EmbeddingManager, Embedder, ImageInput, InputType, SparseEmbedding};
use serde::{Deserialize, Serialize};

use crate::auth::Auth;
use crate::chunk::read_json;
use crate::config::EmbeddingSettings;
use crate::error::ApiError;
use crate::models::{
    EmbeddingLimits, EmbeddingRequest, EmbeddingResponse, EmbeddingStrategiesResponse, EmbeddingStrategyInfo, ErrorBody, SparseVectorOut,
};
use crate::state::AppState;
use crate::worker::{self, JobKind, WorkerFailure};

/// Maior `dimensions` aceito (também é o tamanho do vocabulário/hash nas esparsas).
pub const MAX_DIMENSIONS: usize = 65_536;
/// Imagens: teto de pixels lido do cabeçalho, antes de decodificar (anti-bomba).
const MAX_PIXELS: u64 = 40_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Dense,
    Sparse,
    Multimodal,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Dense => "dense",
            Kind::Sparse => "sparse",
            Kind::Multimodal => "multimodal",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Needs {
    /// Diretório local em `EMBEDDING_MODELS`.
    Model,
    Nothing,
    /// Aprende o vocabulário com os documentos da requisição.
    FittedPerRequest,
    Unsupported(&'static str),
}

struct Spec {
    name: &'static str,
    kind: Kind,
    needs: Needs,
    description: &'static str,
}

const fn spec(name: &'static str, kind: Kind, needs: Needs, description: &'static str) -> Spec {
    Spec { name, kind, needs, description }
}

use Kind::*;
use Needs::*;

/// Uma linha por estratégia do crate `embedding`, na mesma ordem (um teste confere).
const SPECS: &[Spec] = &[
    spec("sentence_transformer", Dense, Model, "Qualquer modelo do sentence-transformers (pooling, Dense e normalização do próprio modelo)."),
    spec("sbert", Dense, Model, "Modelos SBERT clássicos (MiniLM, MPNet)."),
    spec("bge", Dense, Model, "BGE da BAAI (bge-m3 multilíngue, 8192 tokens)."),
    spec("gte", Dense, Model, "GTE da Alibaba (gte-multilingual-base)."),
    spec("e5", Dense, Model, "E5 (multilingual-e5), com os prefixos query:/passage:."),
    spec("multilingual", Dense, Model, "Multilíngues do sentence-transformers (paraphrase-multilingual, LaBSE, distiluse)."),
    spec("instructor", Dense, Model, "INSTRUCTOR: instrução de consulta/documento junto do texto."),
    spec("jina", Dense, Model, "Jina Embeddings v3: adaptador por tarefa (consulta/documento)."),
    spec("nomic", Dense, Model, "Nomic Embed, com os prefixos search_query:/search_document:."),
    spec("huggingface", Dense, Model, "Encoder do Hugging Face com pooling médio."),
    spec("sparse", Sparse, Model, "SPLADE: um peso por termo do vocabulário, com expansão para termos relacionados."),
    spec("tfidf", Sparse, FittedPerRequest, "TF-IDF do scikit-learn, com o vocabulário dos documentos da requisição."),
    spec("hashing", Sparse, Nothing, "Feature hashing do scikit-learn: sem vocabulário, mesmos vetores em qualquer requisição."),
    spec("clip", Multimodal, Model, "CLIP: textos e imagens no mesmo espaço."),
    spec("siglip", Multimodal, Model, "SigLIP: textos e imagens no mesmo espaço."),
    spec("ollama", Dense, Unsupported("o worker de embeddings não tem rede para chamar o Ollama"), "Servidor Ollama local."),
    spec("custom", Dense, Unsupported("embedder injetado: só pela biblioteca"), "Embedder fornecido pelo programa."),
    spec("bm25", Sparse, FittedPerRequest, "Okapi BM25 como vetores: consulta·documento é a nota BM25 (busca híbrida)."),
    spec("lsa", Dense, FittedPerRequest, "TF-IDF + SVD truncado: vetores densos sem modelo, aprendidos com os documentos da requisição."),
];

/// Caminhos que o worker precisa ler para um modelo: o diretório, o caminho real e, se for um
/// snapshot do cache do Hugging Face (links para `../../blobs`), a raiz do repositório no cache.
pub fn model_read_paths(dir: &Path) -> Vec<PathBuf> {
    let mut out = vec![dir.to_path_buf()];
    if let Ok(real) = dir.canonicalize() {
        out.push(real);
    }
    if let Some(snapshots) = dir.parent()
        && snapshots.file_name().is_some_and(|n| n == "snapshots")
        && let Some(repo) = snapshots.parent()
    {
        out.push(repo.to_path_buf());
    }
    out.dedup();
    out
}

/// Catálogo com a disponibilidade DESTE servidor.
pub struct Catalog {
    response: EmbeddingStrategiesResponse,
}

fn display_model(dir: &Path) -> String {
    // Só o nome final: o caminho do servidor não sai na resposta.
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    // Snapshot do cache do Hugging Face: o nome do repositório (`models--dono--nome`).
    match dir.parent().filter(|p| p.file_name().is_some_and(|n| n == "snapshots")).and_then(Path::parent).and_then(Path::file_name) {
        Some(repo) => repo.to_string_lossy().trim_start_matches("models--").replacen("--", "/", 1),
        None => name,
    }
}

impl Catalog {
    pub fn new(settings: &EmbeddingSettings) -> Self {
        let strategies: Vec<EmbeddingStrategyInfo> = SPECS
            .iter()
            .map(|s| {
                let model_dir = settings.models.get(s.name);
                let unavailable = match s.needs {
                    Model if model_dir.is_none() => Some("o servidor não tem modelo local configurado para esta estratégia (EMBEDDING_MODELS)"),
                    Unsupported(reason) => Some(reason),
                    _ => None,
                };
                EmbeddingStrategyInfo {
                    name: s.name.to_string(),
                    aliases: embedding::config::STRATEGY_ALIASES.iter().filter(|(_, c)| *c == s.name).map(|(a, _)| a.to_string()).collect(),
                    description: s.description.to_string(),
                    kind: s.kind.as_str().to_string(),
                    available: unavailable.is_none(),
                    unavailable_reason: unavailable.map(str::to_string),
                    model: model_dir.map(|d| display_model(d)),
                    default_format: if s.kind == Sparse { "sparse" } else { "dense" }.to_string(),
                    accepts_images: s.kind == Multimodal,
                    fitted_per_request: s.needs == FittedPerRequest,
                }
            })
            .collect();
        Self {
            response: EmbeddingStrategiesResponse {
                count: strategies.len(),
                strategies,
                limits: EmbeddingLimits {
                    max_texts: settings.max_texts,
                    max_text_bytes: settings.max_text_bytes,
                    max_images: settings.max_images,
                    max_image_bytes: settings.max_image_bytes,
                    max_dimensions: MAX_DIMENSIONS,
                },
            },
        }
    }

    fn get(&self, name: &str) -> Option<&EmbeddingStrategyInfo> {
        self.response.strategies.iter().find(|s| s.name == name)
    }
}

// ------------------------------------------------------------------ protocolo com o worker

#[derive(Debug, Serialize, Deserialize)]
pub struct Job {
    pub strategy: String,
    pub model_dir: Option<PathBuf>,
    pub device: String,
    pub batch_size: usize,
    pub dimensions: Option<usize>,
    pub normalize: bool,
    pub input_type: InputType,
    pub texts: Vec<String>,
    /// Imagens em base64 (validadas na API).
    pub images: Vec<String>,
    pub queries: Vec<String>,
    pub sparse: bool,
}

/// Motivo de recusa (422). O worker devolve só o código; a mensagem é desta tabela fixa.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    ImageInvalid,
    Failed,
}

impl Rejection {
    pub fn message(self) -> &'static str {
        match self {
            Rejection::ImageInvalid => "imagem inválida, em formato não suportado ou grande demais",
            Rejection::Failed => "não foi possível gerar os embeddings com essa estratégia e esses parâmetros",
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Timings {
    pub confine_ms: u64,
    /// Carga do modelo (tokenizador e sessão ONNX).
    pub load_ms: u64,
    pub embed_ms: u64,
}

/// Vetores densos ou esparsos.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vectors {
    Dense(Vec<Vec<f32>>),
    Sparse(Vec<SparseEmbedding>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Reply {
    Ok {
        dimension: usize,
        vectors: Vectors,
        queries: Option<Vectors>,
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

fn pack(values: Vec<Vec<f32>>, sparse: bool) -> Vectors {
    if sparse { Vectors::Sparse(values.iter().map(|v| SparseEmbedding::from_dense(v)).collect()) } else { Vectors::Dense(values) }
}

fn failed(stage: &str, e: EmbeddingError) -> Result<Reply, String> {
    // Vai para o stderr do worker (log da API); nunca para o cliente.
    eprintln!("{stage}: {e}");
    Ok(Reply::Rejected { reason: Rejection::Failed })
}

/// Executa o job DENTRO do worker (já confinado). `Err` = falha interna.
pub fn execute_job(path: &Path) -> Result<Reply, String> {
    let raw = std::fs::read(path).map_err(|e| format!("leitura do job: {e}"))?;
    let job: Job = serde_json::from_slice(&raw).map_err(|e| format!("job inválido: {e}"))?;
    drop(raw);

    let mut images = Vec::with_capacity(job.images.len());
    for b64 in &job.images {
        let Ok(bytes) = B64.decode(b64) else { return Ok(Reply::Rejected { reason: Rejection::ImageInvalid }) };
        match imagesize::blob_size(&bytes) {
            Ok(d) if (d.width as u64).saturating_mul(d.height as u64) <= MAX_PIXELS => images.push(ImageInput::Bytes(bytes)),
            _ => return Ok(Reply::Rejected { reason: Rejection::ImageInvalid }),
        }
    }

    let mut timings = Timings::default();
    let t = Instant::now();
    let config = EmbeddingConfig {
        strategy: job.strategy.clone(),
        model_name: job.model_dir.as_ref().map(|p| p.to_string_lossy().into_owned()),
        api_key: None,
        device: job.device.clone(),
        batch_size: job.batch_size,
        normalize: job.normalize,
        dimensions: job.dimensions,
        options: Default::default(),
    };
    let manager = match EmbeddingManager::new(config).and_then(|m| m.load().map(|_| m)) {
        Ok(m) => m,
        Err(e) => return failed(&format!("estratégia {}", job.strategy), e),
    };
    timings.load_ms = ms(t.elapsed());

    let t = Instant::now();
    let values = if images.is_empty() {
        let texts: Vec<&str> = job.texts.iter().map(String::as_str).collect();
        manager.embed(&texts, job.input_type)
    } else {
        match manager.embed_images(&images) {
            Err(EmbeddingError::Input(e)) => {
                eprintln!("imagem: {e}");
                return Ok(Reply::Rejected { reason: Rejection::ImageInvalid });
            }
            other => other,
        }
    };
    let values = match values {
        Ok(v) => v,
        Err(e) => return failed(&format!("embedding {}", job.strategy), e),
    };
    let queries = if job.queries.is_empty() {
        None
    } else {
        let q: Vec<&str> = job.queries.iter().map(String::as_str).collect();
        match manager.embed(&q, InputType::Query) {
            Ok(v) => Some(pack(v, job.sparse)),
            Err(e) => return failed(&format!("consultas {}", job.strategy), e),
        }
    };
    timings.embed_ms = ms(t.elapsed());
    let dimension = values.first().map_or(0, Vec::len);
    Ok(Reply::Ok { dimension, vectors: pack(values, job.sparse), queries, timings })
}

// ------------------------------------------------------------------ validação

/// Valida o pedido contra o catálogo e os limites e monta o job (sem ler o conteúdo além de
/// contar bytes e decodificar o base64 das imagens).
fn prepare(req: EmbeddingRequest, settings: &EmbeddingSettings, catalog: &Catalog) -> Result<(Job, String), ApiError> {
    let name = embedding::normalize_strategy(&req.strategy).map_err(|e| match e {
        EmbeddingError::PaidProvider(_) => ApiError::Unprocessable("provedor pago não incluído; use uma estratégia local (veja GET /v1/embedding/strategies)"),
        _ => ApiError::Unprocessable("estratégia desconhecida; veja GET /v1/embedding/strategies"),
    })?;
    let info = catalog.get(name).ok_or(ApiError::Unprocessable("estratégia desconhecida; veja GET /v1/embedding/strategies"))?;
    if !info.available {
        return Err(ApiError::Unprocessable("estratégia indisponível neste servidor; veja GET /v1/embedding/strategies"));
    }
    let input_type: InputType = match req.input_type.as_deref() {
        None => InputType::Document,
        Some(s) => s.parse().map_err(|_| ApiError::Unprocessable("input_type deve ser 'document' ou 'query'"))?,
    };

    let (texts, images) = match (req.texts, req.documents, req.images) {
        (Some(t), None, None) => (t, Vec::new()),
        (None, Some(d), None) => (d.into_iter().map(|d| d.page_content).collect(), Vec::new()),
        (None, None, Some(i)) => (Vec::new(), i),
        _ => return Err(ApiError::BadRequest("envie `texts`, `documents` ou `images` (um deles)")),
    };
    let queries = req.queries.unwrap_or_default();
    if texts.is_empty() && images.is_empty() {
        return Err(ApiError::Unprocessable("a lista de entradas está vazia"));
    }
    if !images.is_empty() && !info.accepts_images {
        return Err(ApiError::Unprocessable("esta estratégia não aceita imagens"));
    }
    if info.fitted_per_request {
        if input_type == InputType::Query {
            return Err(ApiError::Unprocessable(
                "tfidf, bm25 e lsa aprendem o vocabulário dos documentos da requisição: envie o acervo em `texts`/`documents` e as consultas em `queries`",
            ));
        }
    } else if !queries.is_empty() {
        return Err(ApiError::Unprocessable("`queries` só vale para tfidf, bm25 e lsa; nas outras, use input_type 'query'"));
    }
    if texts.len() > settings.max_texts || queries.len() > settings.max_texts {
        return Err(ApiError::Unprocessable("textos demais numa requisição; veja GET /v1/embedding/strategies"));
    }
    let bytes: usize = texts.iter().chain(&queries).map(String::len).sum();
    if bytes > settings.max_text_bytes {
        return Err(ApiError::PayloadTooLarge);
    }
    if images.len() > settings.max_images {
        return Err(ApiError::Unprocessable("imagens demais numa requisição; veja GET /v1/embedding/strategies"));
    }
    let mut image_bytes = 0usize;
    for img in &images {
        let decoded = B64.decode(img).map_err(|_| ApiError::BadRequest("imagem com base64 inválido"))?;
        image_bytes += decoded.len();
    }
    if image_bytes > settings.max_image_bytes {
        return Err(ApiError::PayloadTooLarge);
    }
    if req.dimensions.is_some_and(|d| d == 0 || d > MAX_DIMENSIONS) {
        return Err(ApiError::Unprocessable("dimensions fora do intervalo; veja GET /v1/embedding/strategies"));
    }
    let sparse = match req.format.as_deref() {
        None => info.default_format == "sparse",
        Some("dense") => false,
        Some("sparse") => true,
        Some(_) => return Err(ApiError::Unprocessable("format deve ser 'dense' ou 'sparse'")),
    };

    let job = Job {
        strategy: name.to_string(),
        model_dir: settings.models.get(name).cloned(),
        device: settings.device.clone(),
        batch_size: settings.batch_size,
        dimensions: req.dimensions,
        normalize: req.normalize.unwrap_or(false),
        input_type,
        texts,
        images,
        queries,
        sparse,
    };
    Ok((job, info.model.clone().unwrap_or_else(|| name.to_string())))
}

/// Teto do corpo: textos (com escape do JSON) e imagens em base64 (4/3 do tamanho).
fn body_limit(s: &EmbeddingSettings) -> usize {
    s.max_text_bytes.saturating_mul(2).saturating_add(s.max_image_bytes.saturating_mul(4) / 3).saturating_add(1024 * 1024)
}

fn to_out(v: Vectors) -> (Option<Vec<Vec<f32>>>, Option<Vec<SparseVectorOut>>) {
    match v {
        Vectors::Dense(d) => (Some(d), None),
        Vectors::Sparse(s) => (None, Some(s.into_iter().map(|e| SparseVectorOut { indices: e.indices, values: e.values }).collect())),
    }
}

// ------------------------------------------------------------------ rotas

/// Lista as estratégias de embedding e se cada uma está disponível neste servidor.
///
/// Para cada estratégia: apelidos, tipo (`dense`, `sparse`, `multimodal`), o modelo configurado,
/// o formato padrão da resposta, se aceita imagens e se aprende o vocabulário com os documentos
/// da própria requisição (`tfidf`, `bm25`, `lsa`).
#[utoipa::path(
    get, path = "/v1/embedding/strategies", tag = "Embeddings",
    security(("bearer" = []), ("oauth2" = ["ocr"])),
    responses(
        (status = 200, description = "Estratégias, disponibilidade e limites", body = EmbeddingStrategiesResponse),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 403, description = "Falta o escopo `ocr`", body = ErrorBody),
    )
)]
pub async fn strategies(State(st): State<AppState>, auth: Auth) -> Result<Json<EmbeddingStrategiesResponse>, ApiError> {
    auth.require_scope("ocr")?;
    Ok(Json(st.embedding_catalog.response.clone()))
}

/// Gera embeddings.
///
/// Envie **um** de: `texts` (textos), `documents` (a lista `chunks` de `POST /v1/chunk`, ou
/// qualquer lista de `{page_content, metadata}`; só o texto é usado) ou `images` (base64, só em
/// `clip`/`siglip`). `input_type` escolhe o modo de consulta ou de documento do modelo
/// (prefixos, instruções, tarefas). `dimensions` corta os vetores (Matryoshka; nas esparsas, é o
/// tamanho do vocabulário/hash) e `normalize` os deixa com norma 1.
///
/// `tfidf`, `bm25` e `lsa` aprendem o vocabulário com os documentos desta requisição: envie o
/// acervo em `texts`/`documents` e as consultas em `queries`; os vetores das consultas voltam em
/// `query_embeddings`/`sparse_query_embeddings`.
///
/// Vetores densos voltam em `embeddings`; esparsos (padrão em `sparse`, `tfidf`, `hashing` e
/// `bm25`; escolha com `format`) em `sparse_embeddings`, como índices e valores.
#[utoipa::path(
    post, path = "/v1/embedding", tag = "Embeddings",
    security(("bearer" = []), ("oauth2" = ["ocr"])),
    request_body = EmbeddingRequest,
    responses(
        (status = 200, description = "Um vetor por entrada, na ordem enviada", body = EmbeddingResponse),
        (status = 400, description = "JSON inválido, campo desconhecido, base64 inválido, ou nenhuma/mais de uma entre `texts`, `documents` e `images`", body = ErrorBody),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 403, description = "Falta o escopo `ocr`", body = ErrorBody),
        (status = 413, description = "Textos ou imagens acima do limite", body = ErrorBody),
        (status = 422, description = "Estratégia desconhecida, paga ou indisponível, parâmetro inválido, entradas demais, imagem inválida ou falha do modelo", body = ErrorBody),
        (status = 429, description = "Limite de taxa ou embedding já em andamento para este usuário", body = ErrorBody),
        (status = 503, description = "Servidor ocupado, sem memória ou tempo esgotado", body = ErrorBody),
    )
)]
pub async fn embed(State(st): State<AppState>, auth: Auth, headers: HeaderMap, body: Body) -> Result<Json<EmbeddingResponse>, ApiError> {
    let started = Instant::now();
    auth.require_scope("ocr")?;
    let user = auth.user_id()?;

    // 1) Limites ANTES de ler o corpo.
    st.limiters.check_embedding(user).await.map_err(ApiError::TooManyRequests)?;
    let guard = st.embedding.try_user_guard(user).ok_or(ApiError::TooManyRequests(Duration::from_secs(2)))?;

    // 2) Corpo e validação (barata, na API).
    let cfg = &st.cfg.embedding;
    let req: EmbeddingRequest = read_json(&headers, body, body_limit(cfg)).await?;
    let (job, model) = prepare(req, cfg, &st.embedding_catalog)?;
    let (strategy, input_type, sparse, count) = (job.strategy.clone(), job.input_type, job.sparse, job.texts.len() + job.images.len());
    let model_dir = job.model_dir.clone();

    // 3) Vaga global.
    if st.memory.is_under_pressure() {
        tracing::warn!("embedding recusado: memória acima do limite configurado");
        return Err(ApiError::Unavailable(Duration::from_secs(15)));
    }
    let permit = match tokio::time::timeout(Duration::from_secs(2), st.embedding.slots.clone().acquire_owned()).await {
        Ok(Ok(p)) => p,
        _ => return Err(ApiError::Unavailable(Duration::from_secs(5))),
    };

    // 4) Worker isolado; o job vai num arquivo do diretório exclusivo, apagado no fim.
    let tmp = tempfile::Builder::new().prefix("embed-").tempdir().map_err(|e| ApiError::internal("criar diretório temporário", e))?;
    let path = tmp.path().join("job.json");
    let data = serde_json::to_vec(&job).map_err(|e| ApiError::internal("serializar job", e))?;
    drop(job);
    tokio::fs::write(&path, data).await.map_err(|e| ApiError::internal("gravar job", e))?;
    let t_worker = Instant::now();
    let outcome = worker::run_embedding_job(&st.cfg, tmp.path(), &path, model_dir.as_deref()).await;
    let worker_ms = ms(t_worker.elapsed());
    drop((permit, guard, tmp));

    let (dimension, vectors, queries) = match outcome {
        Ok(Reply::Ok { dimension, vectors, queries, timings: w }) => {
            tracing::info!(
                strategy = %strategy,
                inputs = count,
                worker_ms,
                worker_spawn_ms = worker_ms.saturating_sub(w.confine_ms + w.load_ms + w.embed_ms),
                confine_ms = w.confine_ms,
                load_ms = w.load_ms,
                embed_ms = w.embed_ms,
                total_ms = ms(started.elapsed()),
                "embedding concluído"
            );
            (dimension, vectors, queries)
        }
        Ok(Reply::Rejected { reason }) => {
            if reason == Rejection::Failed {
                tracing::warn!(strategy = %strategy, "embedding falhou");
            }
            return Err(ApiError::Unprocessable(reason.message()));
        }
        Err(WorkerFailure::Timeout) => return Err(ApiError::Unavailable(Duration::from_secs(10))),
        Err(WorkerFailure::ReplyTooLarge) => {
            return Err(ApiError::Unprocessable("resposta grande demais; envie menos entradas, use format 'sparse' ou reduza dimensions"));
        }
        Err(failure) => {
            crate::security::event(&st, failure.event_kind(JobKind::Embedding), Some(user), None, None, Some(&failure.to_string())).await;
            return Err(ApiError::Unprocessable(Rejection::Failed.message()));
        }
    };

    let (embeddings, sparse_embeddings) = to_out(vectors);
    let (query_embeddings, sparse_query_embeddings) = queries.map(to_out).unwrap_or((None, None));
    Ok(Json(EmbeddingResponse {
        strategy,
        model,
        input_type: input_type.as_str().to_string(),
        format: if sparse { "sparse" } else { "dense" }.to_string(),
        dimension,
        count,
        embeddings,
        sparse_embeddings,
        query_embeddings,
        sparse_query_embeddings,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> EmbeddingSettings {
        EmbeddingSettings::default()
    }

    fn req(strategy: &str) -> EmbeddingRequest {
        EmbeddingRequest {
            strategy: strategy.into(),
            input_type: None,
            texts: Some(vec!["texto".into()]),
            documents: None,
            images: None,
            queries: None,
            dimensions: None,
            normalize: None,
            format: None,
        }
    }

    fn unprocessable(r: Result<(Job, String), ApiError>) -> &'static str {
        match r {
            Err(ApiError::Unprocessable(m)) => m,
            Err(other) => panic!("esperava 422, veio {other:?}"),
            Ok(_) => panic!("esperava 422, veio Ok"),
        }
    }

    #[test]
    fn catalog_covers_every_strategy_in_order() {
        let names: Vec<&str> = SPECS.iter().map(|s| s.name).collect();
        assert_eq!(names, embedding::SUPPORTED_STRATEGIES);
        let c = Catalog::new(&settings());
        assert!(c.get("hashing").unwrap().available && c.get("bm25").unwrap().fitted_per_request);
        assert!(!c.get("e5").unwrap().available && !c.get("ollama").unwrap().available && !c.get("custom").unwrap().available);
        assert_eq!(c.get("sparse").unwrap().default_format, "sparse");
        assert!(c.get("clip").unwrap().accepts_images);
    }

    #[test]
    fn validation() {
        let s = EmbeddingSettings { max_texts: 2, max_text_bytes: 20, ..settings() };
        let c = Catalog::new(&s);
        let (job, model) = prepare(req("feature_hashing"), &s, &c).unwrap();
        assert_eq!((job.strategy.as_str(), job.sparse, model.as_str()), ("hashing", true, "hashing"));
        assert!(unprocessable(prepare(req("openai"), &s, &c)).contains("pago"));
        assert!(unprocessable(prepare(req("nope"), &s, &c)).contains("desconhecida"));
        assert!(unprocessable(prepare(req("e5"), &s, &c)).contains("indisponível"));
        let q = EmbeddingRequest { input_type: Some("query".into()), ..req("tfidf") };
        assert!(unprocessable(prepare(q, &s, &c)).contains("acervo"));
        let q = EmbeddingRequest { queries: Some(vec!["x".into()]), ..req("hashing") };
        assert!(unprocessable(prepare(q, &s, &c)).contains("queries"));
        let many = EmbeddingRequest { texts: Some(vec!["a".into(); 3]), ..req("hashing") };
        assert!(unprocessable(prepare(many, &s, &c)).contains("demais"));
        let big = EmbeddingRequest { texts: Some(vec!["a".repeat(21)]), ..req("hashing") };
        assert!(matches!(prepare(big, &s, &c), Err(ApiError::PayloadTooLarge)));
        let img = EmbeddingRequest { texts: None, images: Some(vec!["aGk=".into()]), ..req("hashing") };
        assert!(unprocessable(prepare(img, &s, &c)).contains("imagens"));
        let both = EmbeddingRequest { images: Some(vec![]), ..req("hashing") };
        assert!(matches!(prepare(both, &s, &c), Err(ApiError::BadRequest(_))));
        let dims = EmbeddingRequest { dimensions: Some(0), ..req("hashing") };
        assert!(unprocessable(prepare(dims, &s, &c)).contains("dimensions"));
        let fmt = EmbeddingRequest { format: Some("dense".into()), ..req("hashing") };
        assert!(!prepare(fmt, &s, &c).unwrap().0.sparse);
    }

    #[test]
    fn hf_cache_snapshots_expose_the_repo_root() {
        let p = Path::new("/c/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/abc");
        assert!(model_read_paths(p).contains(&PathBuf::from("/c/hub/models--sentence-transformers--all-MiniLM-L6-v2")));
        assert_eq!(display_model(p), "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(display_model(Path::new("/models/bge-m3")), "bge-m3");
    }
}
