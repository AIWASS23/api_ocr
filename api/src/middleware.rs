use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::error::ApiError;
use crate::limits::client_ip;
use crate::state::AppState;

/// Cabeçalhos de segurança em TODA resposta.
pub async fn security_headers(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let mut resp = next.run(req).await;
    let h = resp.headers_mut();
    let set = |h: &mut axum::http::HeaderMap, k: HeaderName, v: &'static str| {
        h.insert(k, HeaderValue::from_static(v));
    };
    set(h, header::X_CONTENT_TYPE_OPTIONS, "nosniff");
    set(h, header::X_FRAME_OPTIONS, "DENY");
    set(h, header::REFERRER_POLICY, "no-referrer");
    set(h, HeaderName::from_static("cross-origin-resource-policy"), "same-origin");
    set(h, HeaderName::from_static("cross-origin-opener-policy"), "same-origin");
    set(h, HeaderName::from_static("permissions-policy"), "camera=(), microphone=(), geolocation=(), payment=()");
    if st.cfg.secure_cookies() {
        set(h, header::STRICT_TRANSPORT_SECURITY, "max-age=63072000; includeSubDomains");
    }
    // CSP restritiva por padrão; a página de login define a sua.
    if !h.contains_key(header::CONTENT_SECURITY_POLICY) {
        set(h, header::CONTENT_SECURITY_POLICY, "default-src 'none'; frame-ancestors 'none'");
    }
    if !h.contains_key(header::CACHE_CONTROL) {
        set(h, header::CACHE_CONTROL, "no-store");
    }
    resp
}

/// Remove o `X-Request-Id` enviado pelo cliente: o id é sempre gerado aqui.
pub async fn strip_request_id(mut req: Request, next: Next) -> Response {
    req.headers_mut().remove("x-request-id");
    next.run(req).await
}

/// Limite geral por IP.
pub async fn rate_limit_general(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let ip = client_ip(&req, st.cfg.trust_proxy);
    match st.limiters.check_general(ip).await {
        Ok(()) => next.run(req).await,
        Err(wait) => ApiError::TooManyRequests(wait).into_response(),
    }
}

/// Limite estrito por IP nas rotas de autenticação (anti força-bruta / credential stuffing).
pub async fn rate_limit_auth(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let ip = client_ip(&req, st.cfg.trust_proxy);
    match st.limiters.check_auth(ip).await {
        Ok(()) => next.run(req).await,
        Err(wait) => {
            let mut r = ApiError::TooManyRequests(wait).into_response();
            crate::error::no_store(r.headers_mut());
            r
        }
    }
}
