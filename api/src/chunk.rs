//! Chunking: `GET /v1/chunk/strategies` (catálogo) e `POST /v1/chunk` (divide um texto puro ou
//! os `documents` devolvidos por `POST /v1/data_preparation`).
//!
//! O texto vem do cliente e passa por parsers (HTML, JSON, Markdown, regex, tokenizadores). Como
//! no OCR, a divisão roda num processo worker isolado (`worker.rs`): sem segredos, sem rede, com
//! prazo e teto de memória. Um texto patológico (aninhamento que estoura a pilha, explosão de
//! memória) derruba só o worker, nunca a API. Antes disso, a API valida tudo o que é barato:
//! estratégia, tamanhos, quantidade de texto e de documentos, estimativa de chunks. O corpo só é
//! lido depois do limite de taxa e da vaga do usuário.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, header};
use chunking::{ChunkConfig, ChunkManager, Chunker, Document};
use http_body_util::{BodyExt, LengthLimitError, Limited};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::auth::Auth;
use crate::config::ChunkSettings;
use crate::error::ApiError;
use crate::models::{ChunkLimits, ChunkOut, ChunkRequest, ChunkResponse, ErrorBody, OverlapParam, SizeParam, StrategiesResponse, StrategyInfo};
use crate::state::AppState;
use crate::worker::{self, JobKind, WorkerFailure};

/// Maior `chunk_size` aceito (em caracteres ou tokens).
pub const MAX_CHUNK_SIZE: usize = 100_000;
/// Metadados de cada documento de entrada (JSON serializado). Eles são copiados para TODOS os
/// chunks do documento: sem teto, poucos KB virariam centenas de MB na resposta.
const MAX_METADATA_BYTES: usize = 4 * 1024;
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(30);
/// Padrões de tamanho por unidade.
const DEFAULT_CHARS: usize = 1000;
const DEFAULT_TOKENS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Characters,
    Tokens,
}

impl Unit {
    fn as_str(self) -> &'static str {
        match self {
            Unit::Characters => "characters",
            Unit::Tokens => "tokens",
        }
    }
}

/// O que a estratégia precisa além do texto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Needs {
    Nothing,
    /// Tokenizador local (`CHUNK_TOKENIZER`).
    Tokenizer,
    /// Tokenizador e o limite de tokens do modelo (`sentence_bert_config.json` ao lado dele).
    SentenceTransformer,
    /// `punkt_tab` do NLTK; sem ele, a estratégia usa `sentence`.
    Punkt,
    /// Modelo que esta API não carrega (motivo).
    Unsupported(&'static str),
}

struct Spec {
    name: &'static str,
    unit: Unit,
    /// Usa `chunk_overlap`.
    overlap: bool,
    needs: Needs,
    description: &'static str,
}

const fn spec(name: &'static str, unit: Unit, overlap: bool, needs: Needs, description: &'static str) -> Spec {
    Spec { name, unit, overlap, needs, description }
}

use Needs::*;
use Unit::*;

/// Uma linha por estratégia do crate `chunking`, na mesma ordem (um teste confere).
const SPECS: &[Spec] = &[
    spec("character", Characters, true, Nothing, "Tamanho fixo, cortando em qualquer caractere."),
    spec("recursive", Characters, true, Nothing, "Corte recursivo por estrutura: parágrafos, linhas, palavras e, por fim, caracteres. Boa escolha geral."),
    spec("word", Characters, true, Nothing, "Recursivo, preferindo fronteiras de palavra."),
    spec("sentence", Characters, true, Nothing, "Recursivo, preferindo fronteiras de frase."),
    spec("paragraph", Characters, true, Nothing, "Recursivo, preferindo parágrafos (linha em branco)."),
    spec("line", Characters, true, Nothing, "Recursivo, preferindo quebras de linha."),
    spec("token", Tokens, true, Tokenizer, "Recursivo, medido em tokens do tokenizador Hugging Face do servidor."),
    spec("token_tiktoken", Tokens, true, Nothing, "Janelas fixas de tokens do tiktoken (cl100k_base, o dos modelos da OpenAI)."),
    spec("token_sentence_transformer", Tokens, true, SentenceTransformer, "Janelas de tokens limitadas ao contexto do modelo Sentence Transformers do servidor."),
    spec("semchunk", Tokens, true, Tokenizer, "Algoritmo do semchunk: separadores mais significativos primeiro, medido em tokens."),
    spec("semantic", Characters, true, Unsupported("precisa de um modelo de embeddings, que esta API não carrega"), "Corta onde o assunto muda (distância entre embeddings de frases)."),
    spec("nltk", Characters, true, Punkt, "Frases pelo Punkt do NLTK (português), juntadas até o tamanho."),
    spec("spacy", Characters, true, Nothing, "Frases pelo sentencizer do spaCy (português), juntadas até o tamanho."),
    spec("markdown", Characters, true, Nothing, "Recursivo com separadores de Markdown (títulos, blocos de código, réguas)."),
    spec("markdown_header", Characters, true, Nothing, "Uma seção por título Markdown (# a ######); seções grandes são subdivididas."),
    spec("markdown_syntax", Characters, true, Nothing, "Respeita a sintaxe Markdown: títulos, blocos de código inteiros e réguas."),
    spec("html", Characters, true, Nothing, "HTML como código: corta antes de <div>, <p>, <table>, <li>..."),
    spec("html_header", Characters, true, Nothing, "HTML dividido pelos títulos h1 a h6, com o caminho de títulos no texto."),
    spec("html_section", Characters, true, Nothing, "HTML em seções pelos títulos (texto com fonte grande também vira título)."),
    spec("html_semantic", Characters, true, Nothing, "HTML por títulos, mantendo tabelas, listas e código inteiros."),
    spec("json", Characters, true, Nothing, "JSON dividido por chaves, preservando o caminho de cada valor."),
    spec("latex", Characters, true, Nothing, "Seções e ambientes do LaTeX."),
    spec("code", Characters, true, Nothing, "Código-fonte pela linguagem (nesta API, Python): classes, funções, blocos."),
    spec("python", Characters, true, Nothing, "Código Python: classes, funções, métodos, blocos."),
    spec("javascript", Characters, true, Nothing, "JavaScript/TypeScript/JSX/Vue/Svelte, cortando também antes de componentes."),
    spec("markdown_header_contextual", Characters, true, Nothing, "Títulos Markdown com o caminho (Doc > Capítulo > Seção) repetido em cada chunk."),
    spec("docling_hybrid", Tokens, false, Tokenizer, "HybridChunker do docling: um chunk por item (listas e tabelas inteiras), com os títulos; medido em tokens."),
    spec("table", Characters, true, Nothing, "Tabelas Markdown inteiras; as grandes são divididas repetindo o cabeçalho."),
    spec("parent_child", Characters, true, Nothing, "Small-to-big: filhos pequenos para a busca; cada um leva o pai (4x maior) em metadata.parent_content."),
    spec("contextual_retrieval", Characters, true, Unsupported("precisa de um LLM, que esta API não chama"), "Um LLM escreve o contexto de cada chunk antes do texto."),
    spec("legal", Characters, true, Nothing, "Legislação brasileira: um artigo por chunk, com Lei > Título > Capítulo > Seção."),
    spec("qa", Characters, true, Nothing, "FAQ: um par pergunta/resposta por chunk."),
    spec("late_chunking", Characters, true, Unsupported("precisa de um modelo de embeddings por token, que esta API não carrega"), "Embedding de cada chunk calculado com o documento inteiro como contexto."),
];

/// `tokenizer.json` do caminho configurado (o próprio arquivo ou o diretório dele).
pub fn tokenizer_json(path: &Path) -> PathBuf {
    if path.is_dir() { path.join("tokenizer.json") } else { path.to_path_buf() }
}

/// `max_seq_length` do `sentence_bert_config.json` ao lado do tokenizador.
fn max_seq_length(tokenizer: &Path) -> Option<usize> {
    let file = tokenizer_json(tokenizer).with_file_name("sentence_bert_config.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(file).ok()?).ok()?;
    v.get("max_seq_length")?.as_u64().map(|n| n as usize).filter(|&n| n > 0)
}

/// Diretórios onde o worker acha o `punkt_tab`: o configurado e os do sistema (o HOME e a
/// variável `NLTK_DATA` não chegam ao worker).
fn punkt_available(nltk_data: Option<&Path>) -> bool {
    let system = ["/usr/share/nltk_data", "/usr/local/share/nltk_data", "/usr/lib/nltk_data", "/usr/local/lib/nltk_data"];
    nltk_data
        .into_iter()
        .map(Path::to_path_buf)
        .chain(system.iter().map(PathBuf::from))
        .any(|d| d.join("tokenizers/punkt_tab/portuguese/abbrev_types.txt").is_file())
}

/// Catálogo com a disponibilidade DESTE servidor (tokenizador e dados configurados).
pub struct Catalog {
    response: StrategiesResponse,
}

impl Catalog {
    pub fn new(settings: &ChunkSettings) -> Self {
        let tokenizer = settings.tokenizer.as_deref();
        let seq = tokenizer.and_then(max_seq_length);
        let punkt = punkt_available(settings.nltk_data.as_deref());
        let strategies = SPECS
            .iter()
            .map(|s| {
                let (mut default, mut max) = match s.unit {
                    Characters => (DEFAULT_CHARS, MAX_CHUNK_SIZE),
                    Tokens => (DEFAULT_TOKENS, MAX_CHUNK_SIZE),
                };
                let (unavailable, note): (Option<&str>, Option<&str>) = match s.needs {
                    Nothing => (None, None),
                    Tokenizer if tokenizer.is_none() => (Some("o servidor não tem tokenizador configurado (CHUNK_TOKENIZER)"), None),
                    Tokenizer => (None, Some("tokens do tokenizador configurado no servidor")),
                    SentenceTransformer => match (tokenizer, seq) {
                        (None, _) => (Some("o servidor não tem tokenizador configurado (CHUNK_TOKENIZER)"), None),
                        (Some(_), None) => (Some("o tokenizador do servidor não informa o limite de tokens (sentence_bert_config.json)"), None),
                        (Some(_), Some(n)) => {
                            // O tamanho efetivo é o menor entre o pedido e o limite do modelo.
                            max = n;
                            default = default.min(n);
                            (None, Some("tokens do tokenizador configurado no servidor; o chunk_size vai até o limite do modelo"))
                        }
                    },
                    Punkt if punkt => (None, None),
                    Punkt => (None, Some("sem os dados punkt_tab do NLTK neste servidor: usa a estratégia sentence")),
                    Unsupported(reason) => (Some(reason), None),
                };
                StrategyInfo {
                    name: s.name.to_string(),
                    aliases: chunking::config::STRATEGY_ALIASES.iter().filter(|(_, c)| *c == s.name).map(|(a, _)| a.to_string()).collect(),
                    description: s.description.to_string(),
                    available: unavailable.is_none(),
                    unavailable_reason: unavailable.map(str::to_string),
                    note: note.map(str::to_string),
                    chunk_size: SizeParam { unit: s.unit.as_str().to_string(), default, min: 1, max },
                    chunk_overlap: s.overlap.then_some(OverlapParam { default: default / 10, min: 0, max_fraction_of_chunk_size: 0.5 }),
                }
            })
            .collect::<Vec<_>>();
        Self {
            response: StrategiesResponse {
                count: strategies.len(),
                strategies,
                limits: ChunkLimits {
                    max_text_bytes: settings.max_text_bytes,
                    max_documents: settings.max_documents,
                    max_chunks: settings.max_chunks,
                    max_metadata_bytes: MAX_METADATA_BYTES,
                },
            },
        }
    }

    fn get(&self, name: &str) -> Option<&StrategyInfo> {
        self.response.strategies.iter().find(|s| s.name == name)
    }
}

// ------------------------------------------------------------------ protocolo com o worker

/// Job gravado no diretório do job e lido pelo worker.
#[derive(Debug, Serialize, Deserialize)]
pub struct Job {
    pub strategy: String,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub documents: Vec<Document>,
    pub max_chunks: usize,
}

/// Motivo de recusa (422). Como no OCR, o worker devolve só o código e a mensagem vem desta
/// tabela fixa: nem um worker comprometido põe texto arbitrário na resposta.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    TooManyChunks,
    Failed,
}

impl Rejection {
    pub fn message(self) -> &'static str {
        match self {
            Rejection::TooManyChunks => "o texto gera chunks demais; aumente o chunk_size ou envie menos texto",
            Rejection::Failed => "não foi possível dividir o texto com essa estratégia e esses parâmetros",
        }
    }
}

/// Duração das fases DENTRO do worker, em milissegundos (só vão para o log).
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Timings {
    pub confine_ms: u64,
    /// Montagem da estratégia (inclui carregar o tokenizador).
    pub load_ms: u64,
    pub chunk_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Reply {
    Ok {
        chunks: Vec<Document>,
        #[serde(default)]
        timings: Timings,
    },
    Rejected { reason: Rejection },
}

fn ms(d: Duration) -> u64 {
    d.as_millis().try_into().unwrap_or(u64::MAX)
}

/// Monta a estratégia com as opções fixas do servidor (o cliente só escolhe estratégia, tamanho
/// e sobreposição).
fn build(job: &Job, tokenizer: Option<&Path>, nltk_data: Option<&Path>) -> chunking::Result<Arc<dyn Chunker>> {
    let mut options = Map::new();
    match job.strategy.as_str() {
        // Texto do cliente é texto: `<|endoftext|>` vira tokens comuns em vez de recusar tudo.
        "token_tiktoken" => {
            options.insert("disallowed_special".into(), Value::Array(vec![]));
        }
        // Sem um armazenamento de pais, o cliente só usa o pai se ele vier no chunk.
        "parent_child" => {
            options.insert("include_parent_content".into(), Value::Bool(true));
        }
        "nltk" => {
            if let Some(d) = nltk_data {
                options.insert("nltk_data".into(), Value::String(d.to_string_lossy().into_owned()));
            }
        }
        _ => {}
    }
    let mut cfg = ChunkConfig::new(&job.strategy, job.chunk_size, job.chunk_overlap)?;
    cfg.options = options;
    if let Some(t) = tokenizer {
        // `token_sentence_transformer` carrega o tokenizador sozinho, pelo `model_name`.
        cfg.model_name = t.to_string_lossy().into_owned();
    }
    let mut manager = ChunkManager::new(cfg)?;
    if matches!(job.strategy.as_str(), "token" | "semchunk" | "docling_hybrid") {
        let path = tokenizer.ok_or_else(|| chunking::ChunkError::ModelNotLoaded("tokenizador não configurado".into()))?;
        let tok = chunking::models::tokenizer::HfTokenizer::load(&path.to_string_lossy())?;
        manager = manager.with_tokenizer(Arc::new(tok));
    }
    manager.build(None)
}

/// Executa o job DENTRO do worker (já confinado). `Err` = falha interna.
pub fn execute_job(path: &Path, tokenizer: Option<&Path>, nltk_data: Option<&Path>) -> Result<Reply, String> {
    let raw = std::fs::read(path).map_err(|e| format!("leitura do job: {e}"))?;
    let job: Job = serde_json::from_slice(&raw).map_err(|e| format!("job inválido: {e}"))?;
    drop(raw);

    let mut timings = Timings::default();
    let t = Instant::now();
    let chunker = match build(&job, tokenizer, nltk_data) {
        Ok(c) => c,
        Err(e) => {
            // Vai para o stderr do worker (log da API); nunca para o cliente.
            eprintln!("estratégia {}: {e}", job.strategy);
            return Ok(Reply::Rejected { reason: Rejection::Failed });
        }
    };
    timings.load_ms = ms(t.elapsed());
    let t = Instant::now();
    let chunks = match chunker.split_documents(&job.documents) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("chunking {}: {e}", job.strategy);
            return Ok(Reply::Rejected { reason: Rejection::Failed });
        }
    };
    timings.chunk_ms = ms(t.elapsed());
    if chunks.len() > job.max_chunks {
        return Ok(Reply::Rejected { reason: Rejection::TooManyChunks });
    }
    Ok(Reply::Ok { chunks, timings })
}

// ------------------------------------------------------------------ validação

/// Valida o pedido contra o catálogo e os limites e monta o job. Tudo aqui é barato e não lê o
/// conteúdo do texto além de contar bytes e caracteres.
fn prepare(req: ChunkRequest, settings: &ChunkSettings, catalog: &Catalog) -> Result<(Job, Unit, bool), ApiError> {
    let name = chunking::normalize_strategy(&req.strategy)
        .map_err(|_| ApiError::Unprocessable("estratégia desconhecida; veja GET /v1/chunk/strategies"))?;
    let info = catalog.get(name).ok_or(ApiError::Unprocessable("estratégia desconhecida; veja GET /v1/chunk/strategies"))?;
    if !info.available {
        return Err(ApiError::Unprocessable("estratégia indisponível neste servidor; veja GET /v1/chunk/strategies"));
    }
    let unit = if info.chunk_size.unit == Tokens.as_str() { Tokens } else { Characters };

    let size = req.chunk_size.unwrap_or(info.chunk_size.default);
    if !(info.chunk_size.min..=info.chunk_size.max).contains(&size) {
        return Err(ApiError::Unprocessable("chunk_size fora do intervalo da estratégia; veja GET /v1/chunk/strategies"));
    }
    let overlap = match &info.chunk_overlap {
        None => {
            if req.chunk_overlap.is_some_and(|o| o > 0) {
                return Err(ApiError::Unprocessable("chunk_overlap não se aplica a esta estratégia"));
            }
            0
        }
        Some(_) => {
            let o = req.chunk_overlap.unwrap_or(size / 10);
            if o > size / 2 {
                return Err(ApiError::Unprocessable("chunk_overlap pode ser no máximo metade do chunk_size"));
            }
            o
        }
    };

    let documents: Vec<Document> = match (req.text, req.documents) {
        (Some(text), None) => vec![Document::new(text)],
        (None, Some(docs)) => docs.into_iter().map(|d| Document::with_metadata(d.page_content, d.metadata)).collect(),
        _ => return Err(ApiError::BadRequest("envie `text` ou `documents` (um dos dois)")),
    };
    if documents.is_empty() {
        return Err(ApiError::Unprocessable("a lista `documents` está vazia"));
    }
    if documents.len() > settings.max_documents {
        return Err(ApiError::Unprocessable("documentos demais numa requisição; veja GET /v1/chunk/strategies"));
    }
    let bytes: usize = documents.iter().map(|d| d.page_content.len()).sum();
    if bytes > settings.max_text_bytes {
        return Err(ApiError::PayloadTooLarge);
    }
    if documents.iter().any(|d| !d.metadata.is_empty() && serde_json::to_vec(&d.metadata).map_or(usize::MAX, |v| v.len()) > MAX_METADATA_BYTES) {
        return Err(ApiError::Unprocessable("metadata de um documento passa do limite; veja GET /v1/chunk/strategies"));
    }
    if documents.iter().all(|d| d.page_content.trim().is_empty()) {
        return Err(ApiError::Unprocessable("texto vazio"));
    }
    // Estimativa barata antes de gastar um worker: cada chunk tem no máximo `chunk_size`
    // caracteres, então há pelo menos `caracteres / chunk_size` chunks (margem de 2x).
    if unit == Characters {
        let chars: usize = documents.iter().map(|d| d.page_content.chars().count()).sum();
        if chars.div_ceil(size.saturating_mul(2)) > settings.max_chunks {
            return Err(ApiError::Unprocessable(Rejection::TooManyChunks.message()));
        }
    }

    let job = Job { strategy: name.to_string(), chunk_size: size, chunk_overlap: overlap, documents, max_chunks: settings.max_chunks };
    Ok((job, unit, info.chunk_overlap.is_some()))
}

/// Teto do corpo: o texto pode crescer com o escape do JSON; os metadados somam pouco.
fn body_limit(settings: &ChunkSettings) -> usize {
    settings.max_text_bytes.saturating_mul(2).saturating_add(1024 * 1024)
}

/// Lê o corpo JSON com teto de tamanho e prazo. Erros de JSON não ecoam o conteúdo enviado.
pub(crate) async fn read_json<T: DeserializeOwned>(headers: &HeaderMap, body: Body, limit: usize) -> Result<T, ApiError> {
    let is_json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|m| m.trim().eq_ignore_ascii_case("application/json"));
    if !is_json {
        return Err(ApiError::BadRequest("envie o corpo como JSON (Content-Type: application/json)"));
    }
    let bytes = match tokio::time::timeout(UPLOAD_TIMEOUT, Limited::new(body, limit).collect()).await {
        Err(_) => return Err(ApiError::BadRequest("tempo de envio esgotado")),
        Ok(Err(e)) if e.is::<LengthLimitError>() => return Err(ApiError::PayloadTooLarge),
        Ok(Err(_)) => return Err(ApiError::BadRequest("corpo inválido")),
        Ok(Ok(c)) => c.to_bytes(),
    };
    serde_json::from_slice(&bytes).map_err(|_| ApiError::BadRequest("JSON inválido, com campo desconhecido ou de tipo errado"))
}

// ------------------------------------------------------------------ rotas

/// Lista as estratégias de chunking e os parâmetros de cada uma.
///
/// Para cada estratégia: apelidos aceitos, se está disponível neste servidor (e por quê, se não
/// está), a unidade do `chunk_size` (caracteres ou tokens) com padrão e intervalo, e o
/// `chunk_overlap` (padrão e máximo), que é `null` quando a estratégia não usa sobreposição.
#[utoipa::path(
    get, path = "/v1/chunk/strategies", tag = "Chunking",
    security(("bearer" = []), ("oauth2" = ["ocr"])),
    responses(
        (status = 200, description = "Estratégias, parâmetros e limites", body = StrategiesResponse),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 403, description = "Falta o escopo `ocr`", body = ErrorBody),
    )
)]
pub async fn strategies(State(st): State<AppState>, auth: Auth) -> Result<Json<StrategiesResponse>, ApiError> {
    auth.require_scope("ocr")?;
    Ok(Json(st.chunk_catalog.response.clone()))
}

/// Divide um texto em chunks.
///
/// Envie `text` (texto puro) **ou** `documents` (a lista devolvida por `POST /v1/data_preparation`,
/// ou qualquer lista de `{page_content, metadata}`). Escolha a `strategy` (nome ou apelido) e, quando
/// a estratégia usar, `chunk_size` e `chunk_overlap`; os padrões e limites de cada uma estão em
/// `GET /v1/chunk/strategies`. Os metadados de cada documento seguem para os chunks dele, somados
/// aos da estratégia (títulos, artigo de lei, `parent_id`...).
///
/// Cada usuário divide 1 texto por vez; com o servidor ocupado (ou com pouca memória) a resposta é
/// 503 com `Retry-After`.
#[utoipa::path(
    post, path = "/v1/chunk", tag = "Chunking",
    security(("bearer" = []), ("oauth2" = ["ocr"])),
    request_body = ChunkRequest,
    responses(
        (status = 200, description = "Chunks, na ordem do texto", body = ChunkResponse),
        (status = 400, description = "JSON inválido, campo desconhecido, ou nenhum/ambos de `text` e `documents`", body = ErrorBody),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 403, description = "Falta o escopo `ocr`", body = ErrorBody),
        (status = 413, description = "Texto acima do limite", body = ErrorBody),
        (status = 422, description = "Estratégia desconhecida ou indisponível, chunk_size/chunk_overlap fora do intervalo, texto vazio, chunks demais ou texto que a estratégia não consegue dividir", body = ErrorBody),
        (status = 429, description = "Limite de taxa ou chunking já em andamento para este usuário", body = ErrorBody),
        (status = 503, description = "Servidor ocupado, sem memória ou tempo esgotado", body = ErrorBody),
    )
)]
pub async fn chunk(State(st): State<AppState>, auth: Auth, headers: HeaderMap, body: Body) -> Result<Json<ChunkResponse>, ApiError> {
    let started = Instant::now();
    auth.require_scope("ocr")?;
    let user = auth.user_id()?;

    // 1) Limites ANTES de ler o corpo.
    st.limiters.check_chunk(user).await.map_err(ApiError::TooManyRequests)?;
    let guard = st.chunk.try_user_guard(user).ok_or(ApiError::TooManyRequests(Duration::from_secs(2)))?;

    // 2) Corpo e validação (barata, na API).
    let cfg = &st.cfg.chunk;
    let req: ChunkRequest = read_json(&headers, body, body_limit(cfg)).await?;
    let (job, unit, has_overlap) = prepare(req, cfg, &st.chunk_catalog)?;
    let (strategy, chunk_size, chunk_overlap) = (job.strategy.clone(), job.chunk_size, job.chunk_overlap);
    let (documents, bytes) = (job.documents.len(), job.documents.iter().map(|d| d.page_content.len()).sum::<usize>());

    // 3) Vaga global.
    if st.memory.is_under_pressure() {
        tracing::warn!("chunking recusado: memória acima do limite configurado");
        return Err(ApiError::Unavailable(Duration::from_secs(15)));
    }
    let permit = match tokio::time::timeout(Duration::from_secs(2), st.chunk.slots.clone().acquire_owned()).await {
        Ok(Ok(p)) => p,
        _ => return Err(ApiError::Unavailable(Duration::from_secs(5))),
    };

    // 4) Worker isolado. O job vai num arquivo do diretório exclusivo, apagado no fim.
    let tmp = tempfile::Builder::new().prefix("chunk-").tempdir().map_err(|e| ApiError::internal("criar diretório temporário", e))?;
    let path = tmp.path().join("job.json");
    let data = serde_json::to_vec(&job).map_err(|e| ApiError::internal("serializar job", e))?;
    drop(job);
    tokio::fs::write(&path, data).await.map_err(|e| ApiError::internal("gravar job", e))?;
    let t_worker = Instant::now();
    let outcome = worker::run_chunk_job(&st.cfg, tmp.path(), &path).await;
    let worker_ms = ms(t_worker.elapsed());
    drop((permit, guard, tmp));

    let chunks = match outcome {
        Ok(Reply::Ok { chunks, timings: w }) => {
            tracing::info!(
                strategy = %strategy,
                documents,
                bytes,
                chunks = chunks.len(),
                worker_ms,
                worker_spawn_ms = worker_ms.saturating_sub(w.confine_ms + w.load_ms + w.chunk_ms),
                confine_ms = w.confine_ms,
                load_ms = w.load_ms,
                chunk_ms = w.chunk_ms,
                total_ms = ms(started.elapsed()),
                "chunking concluído"
            );
            chunks
        }
        Ok(Reply::Rejected { reason }) => {
            if reason == Rejection::Failed {
                tracing::warn!(strategy = %strategy, "chunking falhou");
            }
            return Err(ApiError::Unprocessable(reason.message()));
        }
        Err(WorkerFailure::Timeout) => return Err(ApiError::Unavailable(Duration::from_secs(10))),
        Err(WorkerFailure::ReplyTooLarge) => return Err(ApiError::Unprocessable(Rejection::TooManyChunks.message())),
        Err(failure) => {
            crate::security::event(&st, failure.event_kind(JobKind::Chunk), Some(user), None, None, Some(&failure.to_string())).await;
            return Err(ApiError::Unprocessable(Rejection::Failed.message()));
        }
    };

    let chunks: Vec<ChunkOut> = chunks.into_iter().map(|d| ChunkOut { page_content: d.page_content, metadata: d.metadata }).collect();
    Ok(Json(ChunkResponse {
        strategy,
        unit: unit.as_str().to_string(),
        chunk_size,
        chunk_overlap: has_overlap.then_some(chunk_overlap),
        count: chunks.len(),
        chunks,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::InputDocument;

    fn req(strategy: &str, size: Option<usize>, overlap: Option<usize>, text: &str) -> ChunkRequest {
        ChunkRequest { strategy: strategy.into(), chunk_size: size, chunk_overlap: overlap, text: Some(text.into()), documents: None }
    }

    fn unprocessable(r: Result<(Job, Unit, bool), ApiError>) -> &'static str {
        match r {
            Err(ApiError::Unprocessable(m)) => m,
            Err(other) => panic!("esperava 422, veio {other:?}"),
            Ok(_) => panic!("esperava 422, veio Ok"),
        }
    }

    #[test]
    fn catalog_covers_every_strategy_in_order() {
        let names: Vec<&str> = SPECS.iter().map(|s| s.name).collect();
        assert_eq!(names, chunking::SUPPORTED_STRATEGIES);
        let c = Catalog::new(&ChunkSettings::default());
        let token = c.get("token").unwrap();
        assert!(!token.available && token.unavailable_reason.as_deref().unwrap().contains("CHUNK_TOKENIZER"));
        assert!(c.get("recursive").unwrap().available);
        assert!(c.get("docling_hybrid").unwrap().chunk_overlap.is_none());
        assert_eq!(c.get("recursive").unwrap().chunk_overlap.as_ref().unwrap().default, 100);
        assert!(c.get("recursive").unwrap().aliases.contains(&"recursive_character".to_string()));
        assert!(!c.get("semantic").unwrap().available);
    }

    #[test]
    fn validation() {
        let s = ChunkSettings { max_chunks: 100, max_text_bytes: 10_000, ..ChunkSettings::default() };
        let c = Catalog::new(&s);

        let (job, unit, overlap) = prepare(req("Fixed-Size", None, None, "abc"), &s, &c).unwrap();
        assert_eq!((job.strategy.as_str(), job.chunk_size, job.chunk_overlap, unit, overlap), ("character", 1000, 100, Characters, true));
        let (job, ..) = prepare(req("recursive", Some(50), None, "abc"), &s, &c).unwrap();
        assert_eq!(job.chunk_overlap, 5, "padrão: 10% do chunk_size");
        assert!(prepare(req("recursive", Some(50), Some(25), "abc"), &s, &c).is_ok());

        assert!(unprocessable(prepare(req("nope", None, None, "abc"), &s, &c)).contains("desconhecida"));
        assert!(unprocessable(prepare(req("token", None, None, "abc"), &s, &c)).contains("indisponível"));
        assert!(unprocessable(prepare(req("recursive", Some(0), None, "abc"), &s, &c)).contains("chunk_size"));
        assert!(unprocessable(prepare(req("recursive", Some(MAX_CHUNK_SIZE + 1), None, "abc"), &s, &c)).contains("chunk_size"));
        assert!(unprocessable(prepare(req("recursive", Some(50), Some(26), "abc"), &s, &c)).contains("metade"));
        assert!(unprocessable(prepare(req("recursive", None, None, " \n "), &s, &c)).contains("vazio"));
        // 1000 caracteres em chunks de 2 => ~500 chunks > 100.
        assert!(unprocessable(prepare(req("recursive", Some(2), Some(0), &"a ".repeat(500)), &s, &c)).contains("chunks demais"));
        assert!(matches!(prepare(req("recursive", None, None, &"a".repeat(10_001)), &s, &c), Err(ApiError::PayloadTooLarge)));

        let both = ChunkRequest { documents: Some(vec![]), ..req("recursive", None, None, "abc") };
        assert!(matches!(prepare(both, &s, &c), Err(ApiError::BadRequest(_))));
        let none = ChunkRequest { text: None, ..req("recursive", None, None, "") };
        assert!(matches!(prepare(none, &s, &c), Err(ApiError::BadRequest(_))));
        let empty = ChunkRequest { text: None, documents: Some(vec![]), ..req("recursive", None, None, "") };
        assert!(unprocessable(prepare(empty, &s, &c)).contains("vazia"));
        let mut big = Map::new();
        big.insert("x".into(), Value::String("y".repeat(MAX_METADATA_BYTES)));
        let meta = ChunkRequest {
            text: None,
            documents: Some(vec![InputDocument { page_content: "abc".into(), metadata: big }]),
            ..req("recursive", None, None, "")
        };
        assert!(unprocessable(prepare(meta, &s, &c)).contains("metadata"));
    }

    #[test]
    fn overlap_only_where_it_applies() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let s = ChunkSettings { tokenizer: Some(dir.path().to_path_buf()), ..ChunkSettings::default() };
        let c = Catalog::new(&s);
        assert!(c.get("docling_hybrid").unwrap().available);
        assert!(unprocessable(prepare(req("hybrid", Some(100), Some(10), "abc"), &s, &c)).contains("não se aplica"));
        let (job, unit, overlap) = prepare(req("hybrid", Some(100), Some(0), "abc"), &s, &c).unwrap();
        assert_eq!((job.chunk_overlap, unit, overlap), (0, Tokens, false));
        // Sem sentence_bert_config.json não há limite do modelo.
        assert!(!c.get("token_sentence_transformer").unwrap().available);
        std::fs::write(dir.path().join("sentence_bert_config.json"), r#"{"max_seq_length": 256}"#).unwrap();
        let c = Catalog::new(&s);
        let st = c.get("token_sentence_transformer").unwrap();
        assert!(st.available);
        assert_eq!((st.chunk_size.default, st.chunk_size.max), (256, 256));
    }
}
