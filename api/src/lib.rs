pub mod auth;
pub mod chunk;
pub mod config;
pub mod crypto;
pub mod docs;
pub mod embed;
pub mod error;
pub mod hardening;
pub mod limits;
pub mod memory;
pub mod mfa;
pub mod middleware;
pub mod models;
pub mod oauth;
pub mod ocr;
pub mod sandbox;
pub mod security;
pub mod server;
pub mod shared_limit;
pub mod state;
pub mod store;
pub mod tokens;
pub mod users;
pub mod validate;
pub mod worker;

use std::str::FromStr;
use std::time::Duration;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderName, HeaderValue, Method, StatusCode, header};
use axum::middleware::{from_fn, from_fn_with_state};
use axum::routing::{get, post};
use axum::{BoxError, Router};
use error::ApiError;
use sqlx::PgPool;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use tower::ServiceBuilder;
use tower::load_shed::error::Overloaded;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::timeout::{RequestBodyTimeoutLayer, TimeoutLayer};
use tower_http::trace::TraceLayer;

use config::Config;
use state::{AppInner, AppState, JobRuntime};

pub static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// Pool com `statement_timeout` e `lock_timeout`: uma query lenta não prende o servidor.
pub async fn connect_db(url: &str) -> anyhow::Result<PgPool> {
    let max = std::env::var("DB_MAX_CONNECTIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    connect_db_with(url, max).await
}

pub async fn connect_db_with(url: &str, max_connections: u32) -> anyhow::Result<PgPool> {
    let opts = PgConnectOptions::from_str(url)?.options([("statement_timeout", "10000"), ("lock_timeout", "5000")]);
    Ok(PgPoolOptions::new()
        .max_connections(max_connections.max(1))
        .acquire_timeout(Duration::from_secs(5))
        .connect_with(opts)
        .await?)
}

pub async fn build_state(cfg: Config, db: PgPool) -> anyhow::Result<AppState> {
    let passwords = crypto::PasswordService::new(cfg.password_pepper.clone(), 4).map_err(|e| anyhow::anyhow!("{e}"))?;
    let tokens = tokens::TokenService::with_previous(&cfg.jwt_secret, &cfg.jwt_verification_secrets()[1..], &cfg.issuer, cfg.access_token_ttl);
    let mut limiters =
        limits::Limiters::new(cfg.rate_general_per_min, cfg.rate_auth_per_min, cfg.rate_ocr_per_min, cfg.rate_chunk_per_min, cfg.rate_embedding_per_min);
    if let Some(url) = &cfg.rate_limit_redis_url {
        // Configurado mas inalcançável na partida = erro de configuração: falha alto e cedo.
        let shared = shared_limit::SharedLimiter::connect(url, &cfg.rate_limit_redis_prefix)
            .await
            .map_err(|e| anyhow::anyhow!("RATE_LIMIT_REDIS_URL inalcançável: {e}"))?;
        tracing::info!("limite de taxa global (Redis) ativo");
        limiters = limiters.with_shared(shared);
    }
    let ocr = JobRuntime::new(cfg.ocr_max_concurrent);
    let chunk = JobRuntime::new(cfg.chunk.max_concurrent);
    let chunk_catalog = chunk::Catalog::new(&cfg.chunk);
    let embedding = JobRuntime::new(cfg.embedding.max_concurrent);
    let embedding_catalog = embed::Catalog::new(&cfg.embedding);
    let memory = memory::MemoryGuard::new(cfg.ocr_memory_watermark_pct);
    let security = security::Monitor::new(cfg.security_alert_login_failures);
    let mfa = mfa::MfaCipher::from_pepper(&cfg.password_pepper);
    Ok(AppState::new(AppInner { cfg, db, passwords, tokens, limiters, ocr, chunk, chunk_catalog, embedding, embedding_catalog, memory, security, mfa }))
}

/// Motor de conversão. Só é criado DENTRO do worker de OCR (ver `worker.rs`).
pub fn ocr_engine(timeout: Duration, max_pages: usize) -> anyhow::Result<data_preparation::DataPreparation> {
    let dp_cfg = data_preparation::DataPreparationConfig {
        document_timeout: Some(timeout),
        max_pages: Some(max_pages),
        ..Default::default()
    };
    Ok(data_preparation::DataPreparation::new(dp_cfg)?)
}

async fn healthz() -> &'static str {
    "ok"
}

/// Liveness: o processo está de pé.
#[utoipa::path(get, path = "/healthz", tag = "Saúde", responses((status = 200, description = "ok", body = String)))]
#[allow(dead_code)]
fn healthz_doc() {}

/// Readiness: o banco responde.
#[utoipa::path(
    get, path = "/readyz", tag = "Saúde",
    responses((status = 200, description = "Pronto"), (status = 503, description = "Banco indisponível"))
)]
#[allow(dead_code)]
fn readyz_doc() {}

async fn readyz(State(st): State<AppState>) -> StatusCode {
    match sqlx::query("SELECT 1").execute(&st.db).await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::SERVICE_UNAVAILABLE,
    }
}

async fn not_found() -> ApiError {
    ApiError::NotFound
}

async fn handle_overload(err: BoxError) -> ApiError {
    if err.is::<Overloaded>() {
        ApiError::Unavailable(Duration::from_secs(2))
    } else {
        ApiError::internal("camada de carga", err)
    }
}

pub fn router(st: AppState) -> Router {
    let cfg = &st.cfg;
    let general_timeout = TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, Duration::from_secs(30));

    // Autenticação: corpo minúsculo + limite estrito por IP.
    let auth_routes = Router::new()
        .route("/oauth/authorize", get(oauth::authorize_get).post(oauth::authorize_post))
        .route("/oauth/token", post(oauth::token))
        .route("/oauth/revoke", post(oauth::revoke))
        .route("/auth/register", post(users::register))
        .layer(DefaultBodyLimit::max(16 * 1024))
        .layer(general_timeout)
        .layer(from_fn_with_state(st.clone(), middleware::rate_limit_auth));

    let api_routes = Router::new()
        .route("/v1/me", get(users::me))
        .layer(DefaultBodyLimit::max(1024))
        .layer(general_timeout);

    // OCR e chunking cuidam dos próprios prazos (envio e processamento) e leem o corpo só depois
    // dos limites por usuário.
    let ocr_routes = Router::new()
        .route("/v1/data_preparation", post(ocr::ocr))
        .layer(DefaultBodyLimit::max(cfg.ocr_max_upload_bytes + 64 * 1024));
    let chunk_routes = Router::new().route("/v1/chunk", post(chunk::chunk));
    let chunk_catalog = Router::new()
        .route("/v1/chunk/strategies", get(chunk::strategies))
        .route("/v1/embedding/strategies", get(embed::strategies))
        .layer(DefaultBodyLimit::max(1024))
        .layer(general_timeout);
    let embedding_routes = Router::new().route("/v1/embedding", post(embed::embed));

    let public = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/.well-known/oauth-authorization-server", get(oauth::metadata))
        .layer(DefaultBodyLimit::max(1024))
        .layer(general_timeout);

    // Documentação: só quando ENABLE_DOCS=true (some inteira, inclusive o /openapi.json, caso contrário).
    let docs_routes = if st.cfg.enable_docs { docs::router(&st) } else { Router::new() };

    let mut app = Router::new()
        .merge(docs_routes)
        .merge(auth_routes)
        .merge(api_routes)
        .merge(ocr_routes)
        .merge(chunk_routes)
        .merge(chunk_catalog)
        .merge(embedding_routes)
        .merge(public)
        .fallback(not_found)
        .with_state(st.clone());

    // CORS só para origens explicitamente liberadas; sem credenciais/cookies.
    if !st.cfg.cors_origins.is_empty() {
        let origins: Vec<HeaderValue> = st.cfg.cors_origins.iter().filter_map(|o| HeaderValue::from_str(o).ok()).collect();
        app = app.layer(
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE])
                .max_age(Duration::from_secs(600)),
        );
    }

    let x_request_id = HeaderName::from_static("x-request-id");
    app.layer(
        ServiceBuilder::new()
            .layer(CatchPanicLayer::new())
            .layer(from_fn(middleware::strip_request_id))
            .layer(SetRequestIdLayer::new(x_request_id.clone(), MakeRequestUuid))
            .layer(
                TraceLayer::new_for_http().make_span_with(|req: &axum::http::Request<_>| {
                    // Só método e caminho: query string pode conter `state`/`code_challenge`.
                    let id = req.headers().get("x-request-id").and_then(|v| v.to_str().ok()).unwrap_or("-");
                    tracing::info_span!("http", method = %req.method(), path = %req.uri().path(), request_id = %id)
                }),
            )
            .layer(PropagateRequestIdLayer::new(x_request_id))
            .layer(from_fn_with_state(st.clone(), middleware::security_headers))
            .layer(from_fn_with_state(st.clone(), middleware::rate_limit_general))
            .layer(RequestBodyTimeoutLayer::new(Duration::from_secs(15)))
            .layer(axum::error_handling::HandleErrorLayer::new(handle_overload))
            .load_shed()
            // GLOBAL: o axum aplica a camada em cada rota; `concurrency_limit()` criaria um
            // semáforo por rota (limite que nunca dispara). Aqui o semáforo é um só.
            .layer(tower::limit::GlobalConcurrencyLimitLayer::new(st.cfg.max_in_flight)),
    )
}

/// Tarefas periódicas: limpeza de registros expirados e poda dos limitadores.
pub fn spawn_maintenance(st: AppState) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(300));
        let mut n = 0u32;
        loop {
            tick.tick().await;
            st.limiters.prune();
            st.security.prune();
            n += 1;
            if n.is_multiple_of(2) {
                match store::cleanup(&st.db).await {
                    Ok(rows) if rows > 0 => tracing::info!(rows, "limpeza de registros expirados"),
                    Ok(_) => {}
                    Err(e) => tracing::warn!(error = %e, "falha na limpeza"),
                }
            }
        }
    });
}
