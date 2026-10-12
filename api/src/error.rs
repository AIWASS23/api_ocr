use std::time::Duration;

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Erros das rotas de negócio (`/v1/*`, `/auth/*`). As mensagens são fixas:
/// nada de detalhes internos, caminhos ou texto vindo do banco.
#[derive(Debug)]
pub enum ApiError {
    BadRequest(&'static str),
    Unauthorized,
    Forbidden(&'static str),
    NotFound,
    Conflict(&'static str),
    PayloadTooLarge,
    Unprocessable(&'static str),
    TooManyRequests(Duration),
    Unavailable(Duration),
    Internal,
}

impl ApiError {
    /// Registra o erro real e devolve uma resposta genérica.
    pub fn internal(context: &str, err: impl std::fmt::Display) -> Self {
        tracing::error!(%context, error = %err, "erro interno");
        ApiError::Internal
    }
}

impl From<sqlx::Error> for ApiError {
    fn from(e: sqlx::Error) -> Self {
        ApiError::internal("banco de dados", e)
    }
}

fn retry_secs(d: Duration) -> u64 {
    d.as_secs().max(1)
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, msg, retry): (StatusCode, &str, &str, Option<u64>) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request", m, None),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "invalid_token",
                "token ausente, inválido ou expirado",
                None,
            ),
            ApiError::Forbidden(m) => (StatusCode::FORBIDDEN, "insufficient_scope", m, None),
            ApiError::NotFound => (StatusCode::NOT_FOUND, "not_found", "recurso não encontrado", None),
            ApiError::Conflict(m) => (StatusCode::CONFLICT, "conflict", m, None),
            ApiError::PayloadTooLarge => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                "corpo da requisição maior que o permitido",
                None,
            ),
            ApiError::Unprocessable(m) => (StatusCode::UNPROCESSABLE_ENTITY, "unprocessable", m, None),
            ApiError::TooManyRequests(d) => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "muitas requisições; tente novamente em instantes",
                Some(retry_secs(d)),
            ),
            ApiError::Unavailable(d) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "unavailable",
                "serviço ocupado; tente novamente em instantes",
                Some(retry_secs(d)),
            ),
            ApiError::Internal => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "erro interno",
                None,
            ),
        };

        let mut resp = (status, Json(json!({ "error": code, "error_description": msg }))).into_response();
        let h = resp.headers_mut();
        if status == StatusCode::UNAUTHORIZED {
            h.insert(
                header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"api-ocr\", error=\"invalid_token\""),
            );
        }
        if status == StatusCode::FORBIDDEN {
            h.insert(
                header::WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"api-ocr\", error=\"insufficient_scope\""),
            );
        }
        if let Some(s) = retry
            && let Ok(v) = HeaderValue::from_str(&s.to_string()) {
                h.insert(header::RETRY_AFTER, v);
            }
        resp
    }
}

/// Erros do endpoint de token, no formato da RFC 6749 §5.2.
#[derive(Debug)]
pub enum OAuthError {
    InvalidRequest(&'static str),
    InvalidClient,
    InvalidGrant,
    UnauthorizedClient,
    UnsupportedGrantType,
    InvalidScope,
    Busy(Duration),
    Server,
}

impl From<sqlx::Error> for OAuthError {
    fn from(e: sqlx::Error) -> Self {
        tracing::error!(error = %e, "erro de banco no endpoint OAuth");
        OAuthError::Server
    }
}

impl IntoResponse for OAuthError {
    fn into_response(self) -> Response {
        let (status, code, desc, retry): (StatusCode, &str, &str, Option<u64>) = match self {
            OAuthError::InvalidRequest(d) => (StatusCode::BAD_REQUEST, "invalid_request", d, None),
            OAuthError::InvalidClient => (StatusCode::UNAUTHORIZED, "invalid_client", "cliente inválido", None),
            // Mensagem única: não revela se o código/token expirou, foi usado ou nunca existiu.
            OAuthError::InvalidGrant => (StatusCode::BAD_REQUEST, "invalid_grant", "concessão inválida ou expirada", None),
            OAuthError::UnauthorizedClient => (StatusCode::BAD_REQUEST, "unauthorized_client", "cliente não autorizado para este fluxo", None),
            OAuthError::UnsupportedGrantType => (StatusCode::BAD_REQUEST, "unsupported_grant_type", "grant_type não suportado", None),
            OAuthError::InvalidScope => (StatusCode::BAD_REQUEST, "invalid_scope", "escopo inválido", None),
            OAuthError::Busy(d) => (StatusCode::SERVICE_UNAVAILABLE, "temporarily_unavailable", "servidor ocupado", Some(retry_secs(d))),
            OAuthError::Server => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", "erro interno", None),
        };
        let mut resp = (status, Json(json!({ "error": code, "error_description": desc }))).into_response();
        no_store(resp.headers_mut());
        if let Some(s) = retry
            && let Ok(v) = HeaderValue::from_str(&s.to_string()) {
                resp.headers_mut().insert(header::RETRY_AFTER, v);
            }
        resp
    }
}

pub fn no_store(h: &mut axum::http::HeaderMap) {
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert(header::PRAGMA, HeaderValue::from_static("no-cache"));
}
