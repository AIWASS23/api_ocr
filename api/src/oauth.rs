//! Servidor OAuth2: Authorization Code + PKCE (S256), refresh token com rotação
//! e detecção de reutilização, revogação (RFC 7009) e metadados (RFC 8414).

use axum::Json;
use axum::extract::{Form, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::{IntoParams, ToSchema};
use sha2::{Digest, Sha256};

use crate::crypto::{self, HashError};
use crate::error::{OAuthError, no_store};
use crate::models::{AuthServerMetadata, ErrorBody, LoginForm, OAuthErrorBody};
use crate::limits::{client_ip_from, origin_key};
use crate::security;
use crate::state::AppState;
use crate::store::{self, CodeOutcome, RotateOutcome};
use crate::validate;

const MAX_FIELD: usize = 2048;

fn csrf_cookie_name(st: &AppState) -> &'static str {
    if st.cfg.secure_cookies() { "__Host-csrf" } else { "csrf" }
}

// ---------------------------------------------------------------- /authorize

#[derive(Debug, Default, Deserialize, Clone, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct AuthorizeParams {
    /// Sempre `code`.
    pub response_type: Option<String>,
    /// Cliente registrado (`api_ocr create-client`).
    pub client_id: Option<String>,
    /// Deve ser IGUAL a uma das URIs registradas para o cliente.
    pub redirect_uri: Option<String>,
    /// Escopos separados por espaço (`ocr`, `profile`). Padrão: todos os do cliente.
    pub scope: Option<String>,
    /// Valor opaco devolvido no redirect (proteção CSRF do cliente).
    pub state: Option<String>,
    /// PKCE: base64url(SHA-256(code_verifier)), 43 caracteres.
    pub code_challenge: Option<String>,
    /// Só `S256` é aceito.
    pub code_challenge_method: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AuthorizeForm {
    #[serde(flatten)]
    params: AuthorizeParams,
    email: Option<String>,
    password: Option<String>,
    /// Código do app autenticador (só para contas com MFA).
    otp: Option<String>,
    csrf: Option<String>,
}

/// Resultado da validação do pedido de autorização.
enum Checked {
    Ok { client: store::ClientRow, redirect_uri: String, scope: String, challenge: String, state: Option<String> },
    /// `client_id`/`redirect_uri` inválidos: NUNCA redireciona (evita open redirect).
    Page(&'static str),
    /// Cliente e redirect_uri válidos, mas outro parâmetro não: devolve o erro ao cliente.
    Redirect { redirect_uri: String, state: Option<String>, error: &'static str },
}

fn too_long(p: &AuthorizeParams) -> bool {
    [&p.response_type, &p.client_id, &p.redirect_uri, &p.scope, &p.code_challenge, &p.code_challenge_method]
        .iter()
        .any(|v| v.as_ref().is_some_and(|s| s.len() > MAX_FIELD))
        || p.state.as_ref().is_some_and(|s| s.len() > 512)
}

async fn check_authorize(st: &AppState, p: &AuthorizeParams) -> Result<Checked, OAuthError> {
    if too_long(p) {
        return Ok(Checked::Page("Pedido inválido."));
    }
    let (Some(client_id), Some(redirect_uri)) = (p.client_id.as_deref(), p.redirect_uri.as_deref()) else {
        return Ok(Checked::Page("Pedido inválido."));
    };
    if !validate::valid_client_id(client_id) {
        return Ok(Checked::Page("Cliente desconhecido."));
    }
    let Some(client) = store::get_client(&st.db, client_id).await? else {
        return Ok(Checked::Page("Cliente desconhecido."));
    };
    // Comparação EXATA com as URIs registradas.
    if !client.is_active || !client.redirect_uris.iter().any(|u| u == redirect_uri) {
        return Ok(Checked::Page("Cliente desconhecido."));
    }

    let state = p.state.clone();
    let fail = |error| Checked::Redirect { redirect_uri: redirect_uri.to_string(), state: state.clone(), error };

    if p.response_type.as_deref() != Some("code") {
        return Ok(fail("unsupported_response_type"));
    }
    // Só S256: `plain` não é aceito.
    if p.code_challenge_method.as_deref() != Some("S256") {
        return Ok(fail("invalid_request"));
    }
    let Some(challenge) = p.code_challenge.as_deref().filter(|c| validate::valid_code_challenge(c)) else {
        return Ok(fail("invalid_request"));
    };
    let requested = match p.scope.as_deref() {
        None | Some("") => client.allowed_scopes.clone(),
        Some(s) => match validate::parse_scopes(s) {
            Some(v) => v,
            None => return Ok(fail("invalid_scope")),
        },
    };
    if requested.is_empty() || !requested.iter().all(|s| client.allowed_scopes.contains(s) && validate::SUPPORTED_SCOPES.contains(&s.as_str())) {
        return Ok(fail("invalid_scope"));
    }
    let mut scope = requested;
    scope.sort();

    Ok(Checked::Ok {
        redirect_uri: redirect_uri.to_string(),
        scope: scope.join(" "),
        challenge: challenge.to_string(),
        state: state.clone(),
        client,
    })
}

fn redirect_with(redirect_uri: &str, issuer: &str, pairs: &[(&str, &str)], state: Option<&str>) -> Response {
    let Ok(mut url) = url::Url::parse(redirect_uri) else {
        return error_page(StatusCode::BAD_REQUEST, "Pedido inválido.");
    };
    {
        let mut q = url.query_pairs_mut();
        for (k, v) in pairs {
            q.append_pair(k, v);
        }
        if let Some(s) = state {
            q.append_pair("state", s);
        }
        q.append_pair("iss", issuer); // RFC 9207
    }
    let mut resp = Redirect::to(url.as_str()).into_response();
    no_store(resp.headers_mut());
    resp
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            _ => o.push(c),
        }
    }
    o
}

const CSS: &str = "body{font-family:system-ui,sans-serif;max-width:22rem;margin:4rem auto;padding:0 1rem;color:#111}\
label{display:block;margin:.8rem 0 .2rem}input{width:100%;padding:.5rem;box-sizing:border-box}\
button{margin-top:1rem;padding:.6rem 1rem;width:100%}.err{color:#b00020}";

fn css_hash() -> String {
    B64.encode(Sha256::digest(CSS.as_bytes()))
}

fn html_response(status: StatusCode, body: String, form_action_origin: Option<&str>) -> Response {
    let action = form_action_origin.map(|o| format!(" {o}")).unwrap_or_default();
    let csp = format!(
        "default-src 'none'; style-src 'sha256-{}'; form-action 'self'{action}; frame-ancestors 'none'; base-uri 'none'",
        css_hash()
    );
    let mut resp = (status, Html(body)).into_response();
    let h = resp.headers_mut();
    no_store(h);
    if let Ok(v) = HeaderValue::from_str(&csp) {
        h.insert(header::CONTENT_SECURITY_POLICY, v);
    }
    resp
}

fn error_page(status: StatusCode, msg: &str) -> Response {
    let body = format!(
        "<!doctype html><html lang=\"pt-BR\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Erro</title><style>{CSS}</style><h1>Erro</h1><p>{}</p></html>",
        esc(msg)
    );
    html_response(status, body, None)
}

fn origin_of(uri: &str) -> Option<String> {
    let u = url::Url::parse(uri).ok()?;
    Some(u.origin().ascii_serialization()).filter(|o| o != "null")
}

fn login_page(client_name: &str, p: &AuthorizeParams, csrf: &str, error: Option<&str>, status: StatusCode) -> Response {
    let hidden = |name: &str, val: &Option<String>| match val {
        Some(v) => format!("<input type=\"hidden\" name=\"{name}\" value=\"{}\">", esc(v)),
        None => String::new(),
    };
    let err = error.map(|e| format!("<p class=\"err\" role=\"alert\">{}</p>", esc(e))).unwrap_or_default();
    let body = format!(
        "<!doctype html><html lang=\"pt-BR\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">\
         <meta name=\"referrer\" content=\"no-referrer\"><title>Entrar</title><style>{CSS}</style>\
         <h1>Entrar</h1><p>Autorizar o acesso de <strong>{name}</strong>.</p>{err}\
         <form method=\"post\" action=\"/oauth/authorize\" autocomplete=\"on\">\
         {h1}{h2}{h3}{h4}{h5}{h6}{h7}<input type=\"hidden\" name=\"csrf\" value=\"{csrf}\">\
         <label for=\"email\">E-mail</label><input id=\"email\" name=\"email\" type=\"email\" required maxlength=\"254\" autocomplete=\"username\">\
         <label for=\"password\">Senha</label><input id=\"password\" name=\"password\" type=\"password\" required maxlength=\"128\" autocomplete=\"current-password\">\
         <label for=\"otp\">Código de verificação (se ativado)</label><input id=\"otp\" name=\"otp\" type=\"text\" inputmode=\"numeric\" pattern=\"[0-9]{{6}}\" maxlength=\"6\" autocomplete=\"one-time-code\">\
         <button type=\"submit\">Entrar</button></form></html>",
        name = esc(client_name),
        h1 = hidden("response_type", &p.response_type),
        h2 = hidden("client_id", &p.client_id),
        h3 = hidden("redirect_uri", &p.redirect_uri),
        h4 = hidden("scope", &p.scope),
        h5 = hidden("state", &p.state),
        h6 = hidden("code_challenge", &p.code_challenge),
        h7 = hidden("code_challenge_method", &p.code_challenge_method),
        csrf = esc(csrf),
    );
    let origin = p.redirect_uri.as_deref().and_then(origin_of);
    html_response(status, body, origin.as_deref())
}

fn csrf_cookie(st: &AppState, value: String) -> Cookie<'static> {
    Cookie::build((csrf_cookie_name(st), value))
        .http_only(true)
        .secure(st.cfg.secure_cookies())
        .same_site(SameSite::Strict)
        .path("/")
        .max_age(time::Duration::minutes(10))
        .build()
}

/// Página de login do fluxo Authorization Code + PKCE.
///
/// Abra no navegador. Se `client_id` ou `redirect_uri` forem inválidos, mostra erro e **nunca**
/// redireciona. Erros nos demais parâmetros voltam ao cliente em `redirect_uri?error=...`.
#[utoipa::path(
    get, path = "/oauth/authorize", tag = "OAuth2",
    params(AuthorizeParams),
    responses(
        (status = 200, description = "Página HTML de login (define o cookie CSRF)", content_type = "text/html", body = String),
        (status = 303, description = "Erro de parâmetro devolvido ao cliente via `redirect_uri?error=...`"),
        (status = 400, description = "client_id/redirect_uri inválidos (sem redirect)", content_type = "text/html", body = String),
        (status = 429, description = "Limite de taxa excedido", body = ErrorBody),
    )
)]
pub async fn authorize_get(State(st): State<AppState>, jar: CookieJar, Query(p): Query<AuthorizeParams>) -> Response {
    match check_authorize(&st, &p).await {
        Err(_) => error_page(StatusCode::INTERNAL_SERVER_ERROR, "Erro interno."),
        Ok(Checked::Page(m)) => error_page(StatusCode::BAD_REQUEST, m),
        Ok(Checked::Redirect { redirect_uri, state, error }) => {
            redirect_with(&redirect_uri, &st.cfg.issuer, &[("error", error)], state.as_deref())
        }
        Ok(Checked::Ok { client, .. }) => {
            let csrf = crypto::random_token(32);
            let jar = jar.add(csrf_cookie(&st, csrf.clone()));
            (jar, login_page(&client.name, &p, &csrf, None, StatusCode::OK)).into_response()
        }
    }
}

/// Envia as credenciais da página de login.
///
/// Em sucesso responde 303 para `redirect_uri?code=...&state=...&iss=...`. Senha errada, usuário
/// inexistente e conta bloqueada produzem a MESMA resposta (401). Exige o cookie CSRF.
#[utoipa::path(
    post, path = "/oauth/authorize", tag = "OAuth2",
    request_body(content = LoginForm, content_type = "application/x-www-form-urlencoded"),
    responses(
        (status = 303, description = "Login ok: redireciona com `code`"),
        (status = 401, description = "E-mail ou senha incorretos, ou conta temporariamente bloqueada", content_type = "text/html", body = String),
        (status = 403, description = "CSRF ausente ou inválido", content_type = "text/html", body = String),
        (status = 429, description = "Limite de taxa excedido", body = ErrorBody),
    )
)]
pub async fn authorize_post(State(st): State<AppState>, jar: CookieJar, req_ip: ClientIp, Form(f): Form<AuthorizeForm>) -> Response {
    // 1) CSRF: cookie (HttpOnly, SameSite=Strict) == campo oculto, em tempo constante.
    let cookie_val = jar.get(csrf_cookie_name(&st)).map(|c| c.value().to_string());
    let form_val = f.csrf.clone().unwrap_or_default();
    let csrf_ok = cookie_val.as_deref().is_some_and(|c| !c.is_empty() && crypto::ct_eq(c.as_bytes(), form_val.as_bytes()));
    if !csrf_ok {
        return error_page(StatusCode::FORBIDDEN, "Sessão expirada. Volte ao aplicativo e tente novamente.");
    }

    // 2) Revalida o pedido inteiro (os campos ocultos vêm do navegador, não são confiáveis).
    let (client, redirect_uri, scope, challenge, state) = match check_authorize(&st, &f.params).await {
        Err(_) => return error_page(StatusCode::INTERNAL_SERVER_ERROR, "Erro interno."),
        Ok(Checked::Page(m)) => return error_page(StatusCode::BAD_REQUEST, m),
        Ok(Checked::Redirect { redirect_uri, state, error }) => {
            return redirect_with(&redirect_uri, &st.cfg.issuer, &[("error", error)], state.as_deref());
        }
        Ok(Checked::Ok { client, redirect_uri, scope, challenge, state }) => (client, redirect_uri, scope, challenge, state),
    };

    let ip = req_ip.0.to_string();
    let origin = origin_key(req_ip.0);
    let again = |msg: &str, status: StatusCode| login_page(&client.name, &f.params, &form_val, Some(msg), status);
    const GENERIC: &str = "E-mail, senha ou código incorretos, ou acesso temporariamente bloqueado.";
    let cid = Some(client.client_id.as_str());

    // 3) Credenciais.
    let email = f.email.as_deref().and_then(validate::normalize_email);
    let password = f.password.unwrap_or_default();
    let Some(email) = email else {
        return again(GENERIC, StatusCode::UNAUTHORIZED);
    };
    if password.is_empty() || password.chars().count() > 128 {
        return again(GENERIC, StatusCode::UNAUTHORIZED);
    }

    let user = match store::find_user_by_email(&st.db, &email).await {
        Ok(u) => u,
        Err(e) => {
            tracing::error!(error = %e, "falha ao buscar usuário");
            return error_page(StatusCode::INTERNAL_SERVER_ERROR, "Erro interno.");
        }
    };

    // Bloqueio por conta+origem (e da conta inteira só para origens desconhecidas). Bloqueado: a
    // senha nem é conferida, mas o hash falso roda do mesmo jeito para o tempo não denunciar.
    if let Some(u) = user.as_ref() {
        let gate = match store::login_gate(&st.db, u.id, &origin).await {
            Ok(g) => g,
            Err(e) => {
                tracing::error!(error = %e, "falha ao consultar bloqueio");
                return error_page(StatusCode::INTERNAL_SERVER_ERROR, "Erro interno.");
            }
        };
        if gate.blocked() {
            let _ = st.passwords.verify(password, None).await;
            let kind = if gate.origin_locked { "login_ip_locked" } else { "login_blocked" };
            security::event(&st, kind, Some(u.id), cid, Some(&ip), None).await;
            return again(GENERIC, StatusCode::UNAUTHORIZED);
        }
    }

    let phc = user.as_ref().filter(|u| u.is_active).map(|u| u.password_hash.clone());
    let password_ok = match st.passwords.verify(password, phc).await {
        Ok(ok) => ok,
        Err(HashError::Busy) => return error_page(StatusCode::SERVICE_UNAVAILABLE, "Servidor ocupado. Tente novamente."),
        Err(HashError::Internal) => return error_page(StatusCode::INTERNAL_SERVER_ERROR, "Erro interno."),
    };

    // Segundo fator, se a conta tiver. Só é conferido com a senha certa (não vira oráculo).
    let mut failure_kind = "login_failed";
    let ok = match user.as_ref() {
        Some(u) if password_ok => match u.totp_secret_enc.as_deref() {
            None => true,
            Some(blob) => {
                let ok = check_totp(&st, u.id, blob, f.otp.as_deref().unwrap_or("")).await;
                if !ok {
                    failure_kind = "mfa_failed";
                }
                ok
            }
        },
        _ => false,
    };

    let Some(user) = user.as_ref().filter(|_| ok) else {
        if let Some(u) = user.as_ref() {
            match store::register_login_failure(&st.db, u.id, &origin).await {
                Ok(out) if out.account_just_locked => {
                    let detail = format!("{} falhas de várias origens em 24 h", store::ACCOUNT_MAX_FAILED_LOGINS);
                    security::event(&st, "account_locked", Some(u.id), cid, Some(&ip), Some(&detail)).await;
                }
                Ok(_) => {}
                Err(e) => tracing::error!(error = %e, "falha ao registrar tentativa de login"),
            }
            security::event(&st, failure_kind, Some(u.id), cid, Some(&ip), None).await;
        } else {
            security::event(&st, "login_failed_unknown", None, cid, Some(&ip), None).await;
        }
        return again(GENERIC, StatusCode::UNAUTHORIZED);
    };

    // 4) Sucesso: emite o código (60 s, uso único).
    if let Err(e) = store::register_login_success(&st.db, user.id, &origin).await {
        tracing::error!(error = %e, "falha ao registrar login");
    }
    let code = crypto::new_opaque_token();
    if let Err(e) = store::insert_auth_code(
        &st.db,
        &crypto::hash_token(&code),
        &client.client_id,
        user.id,
        &redirect_uri,
        &scope,
        &challenge,
        st.cfg.auth_code_ttl.as_secs_f64(),
    )
    .await
    {
        tracing::error!(error = %e, "falha ao gravar authorization code");
        return error_page(StatusCode::INTERNAL_SERVER_ERROR, "Erro interno.");
    }
    security::event(&st, "login_ok", Some(user.id), cid, Some(&ip), None).await;

    // O cookie CSRF cumpriu sua função: remove.
    let jar = jar.remove(Cookie::build(csrf_cookie_name(&st)).path("/").build());
    (jar, redirect_with(&redirect_uri, &st.cfg.issuer, &[("code", &code)], state.as_deref())).into_response()
}

// ---------------------------------------------------------------- /token

#[derive(Debug, Deserialize, ToSchema)]
pub struct TokenForm {
    /// `authorization_code` ou `refresh_token`.
    grant_type: Option<String>,
    /// (authorization_code) Código recebido no redirect.
    code: Option<String>,
    /// (authorization_code) A MESMA redirect_uri usada no /authorize.
    redirect_uri: Option<String>,
    client_id: Option<String>,
    /// (authorization_code) PKCE: o segredo cujo SHA-256 foi enviado como code_challenge.
    code_verifier: Option<String>,
    /// (refresh_token) Token de refresh atual. É de uso único: cada resposta traz um novo.
    refresh_token: Option<String>,
    /// (refresh_token) Opcional; só pode restringir os escopos já concedidos.
    scope: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct TokenResponse {
    /// JWT (HS256) de vida curta, para `Authorization: Bearer`.
    access_token: String,
    #[schema(example = "Bearer")]
    token_type: &'static str,
    /// Segundos até o access token expirar.
    expires_in: u64,
    /// Novo refresh token (o anterior deixa de valer).
    refresh_token: String,
    scope: String,
}

fn token_json(access_token: String, refresh_token: String, scope: String, expires_in: u64) -> Response {
    let mut resp = Json(TokenResponse { access_token, token_type: "Bearer", expires_in, refresh_token, scope }).into_response();
    no_store(resp.headers_mut());
    resp
}

fn opt(v: &Option<String>) -> Result<Option<&str>, OAuthError> {
    match v.as_deref() {
        Some(s) if s.len() > MAX_FIELD => Err(OAuthError::InvalidRequest("parâmetro grande demais")),
        other => Ok(other),
    }
}

/// Troca um código ou um refresh token por tokens.
///
/// - `authorization_code`: exige `code`, `redirect_uri`, `client_id` e `code_verifier` (PKCE).
/// - `refresh_token`: rotaciona o token. **Reapresentar um refresh já usado revoga toda a sessão.**
#[utoipa::path(
    post, path = "/oauth/token", tag = "OAuth2",
    request_body(content = TokenForm, content_type = "application/x-www-form-urlencoded"),
    responses(
        (status = 200, description = "Tokens emitidos", body = TokenResponse),
        (status = 400, description = "invalid_request, invalid_grant, unsupported_grant_type, invalid_scope", body = OAuthErrorBody),
        (status = 401, description = "invalid_client", body = OAuthErrorBody),
        (status = 413, description = "Corpo grande demais", body = ErrorBody),
        (status = 429, description = "Limite de taxa excedido", body = ErrorBody),
        (status = 503, description = "Servidor ocupado", body = OAuthErrorBody),
    )
)]
pub async fn token(State(st): State<AppState>, req_ip: ClientIp, Form(f): Form<TokenForm>) -> Result<Response, OAuthError> {
    let ip = req_ip.0.to_string();
    let client_id = opt(&f.client_id)?.filter(|c| validate::valid_client_id(c)).ok_or(OAuthError::InvalidClient)?;
    let client = store::get_client(&st.db, client_id)
        .await?
        .filter(|c| c.is_active)
        .ok_or(OAuthError::InvalidClient)?;

    match opt(&f.grant_type)? {
        Some("authorization_code") => grant_code(&st, &f, &client, &ip).await,
        Some("refresh_token") => grant_refresh(&st, &f, &client, &ip).await,
        Some(_) => Err(OAuthError::UnsupportedGrantType),
        None => Err(OAuthError::InvalidRequest("grant_type ausente")),
    }
}

async fn grant_code(st: &AppState, f: &TokenForm, client: &store::ClientRow, ip: &str) -> Result<Response, OAuthError> {
    let code = opt(&f.code)?.ok_or(OAuthError::InvalidRequest("code ausente"))?;
    let redirect_uri = opt(&f.redirect_uri)?.ok_or(OAuthError::InvalidRequest("redirect_uri ausente"))?;
    let verifier = opt(&f.code_verifier)?.ok_or(OAuthError::InvalidRequest("code_verifier ausente"))?;
    if !validate::valid_code_verifier(verifier) {
        return Err(OAuthError::InvalidGrant);
    }

    let row = match store::consume_auth_code(&st.db, &crypto::hash_token(code)).await? {
        CodeOutcome::Valid(r) => r,
        CodeOutcome::Reused { user_id, client_id } => {
            // Código já trocado: trata como roubo e derruba os tokens emitidos (refresh E access).
            store::revoke_user_client(&st.db, user_id, &client_id).await?;
            store::invalidate_access_tokens(&st.db, user_id).await?;
            security::event(st, "auth_code_reuse", Some(user_id), Some(&client_id), Some(ip), Some("authorization code reapresentado")).await;
            return Err(OAuthError::InvalidGrant);
        }
        CodeOutcome::Unknown => return Err(OAuthError::InvalidGrant),
    };

    if row.expired
        || row.client_id != client.client_id
        || !crypto::ct_eq(row.redirect_uri.as_bytes(), redirect_uri.as_bytes())
        || !crypto::pkce_matches(verifier, &row.code_challenge)
    {
        security::event(st, "token_exchange_failed", Some(row.user_id), Some(&client.client_id), Some(ip), None).await;
        return Err(OAuthError::InvalidGrant);
    }
    if store::find_email_by_id(&st.db, row.user_id).await?.is_none() {
        return Err(OAuthError::InvalidGrant);
    }

    let access = st.tokens.issue(row.user_id, &client.client_id, &row.scope).map_err(|e| {
        tracing::error!(error = %e, "falha ao assinar JWT");
        OAuthError::Server
    })?;
    let refresh = crypto::new_opaque_token();
    store::insert_refresh_new_family(
        &st.db,
        &crypto::hash_token(&refresh),
        row.user_id,
        &client.client_id,
        &row.scope,
        st.cfg.refresh_token_ttl.as_secs_f64(),
        st.cfg.refresh_family_max.as_secs_f64(),
    )
    .await?;
    security::event(st, "token_issued", Some(row.user_id), Some(&client.client_id), Some(ip), None).await;
    Ok(token_json(access, refresh, row.scope, st.tokens.ttl().as_secs()))
}

async fn grant_refresh(st: &AppState, f: &TokenForm, client: &store::ClientRow, ip: &str) -> Result<Response, OAuthError> {
    let presented = opt(&f.refresh_token)?.ok_or(OAuthError::InvalidRequest("refresh_token ausente"))?;
    if presented.len() != 43 {
        return Err(OAuthError::InvalidGrant); // formato que nunca emitimos
    }
    let requested = match opt(&f.scope)? {
        None | Some("") => None,
        Some(s) => Some(validate::parse_scopes(s).ok_or(OAuthError::InvalidScope)?.join(" ")),
    };

    let new_refresh = crypto::new_opaque_token();
    let outcome = store::rotate_refresh(
        &st.db,
        &crypto::hash_token(presented),
        &client.client_id,
        &crypto::hash_token(&new_refresh),
        requested.as_deref(),
        st.cfg.refresh_token_ttl.as_secs_f64(),
        st.cfg.refresh_family_max.as_secs_f64(),
    )
    .await?;

    match outcome {
        RotateOutcome::Rotated { user_id, scope } => {
            let access = st.tokens.issue(user_id, &client.client_id, &scope).map_err(|e| {
                tracing::error!(error = %e, "falha ao assinar JWT");
                OAuthError::Server
            })?;
            security::event(st, "token_refreshed", Some(user_id), Some(&client.client_id), Some(ip), None).await;
            Ok(token_json(access, new_refresh, scope, st.tokens.ttl().as_secs()))
        }
        RotateOutcome::InvalidScope => Err(OAuthError::InvalidScope),
        RotateOutcome::Invalid => Err(OAuthError::InvalidGrant),
        RotateOutcome::Reuse { user_id } => {
            // A família já foi revogada; os access tokens emitidos a partir dela também caem.
            store::invalidate_access_tokens(&st.db, user_id).await?;
            security::event(st, "refresh_reuse_detected", Some(user_id), Some(&client.client_id), Some(ip), Some("refresh token reapresentado: sessão derrubada")).await;
            Err(OAuthError::InvalidGrant)
        }
    }
}

// ---------------------------------------------------------------- /revoke

#[derive(Debug, Deserialize, ToSchema)]
pub struct RevokeForm {
    /// Refresh token a revogar (revoga a família inteira).
    token: Option<String>,
    client_id: Option<String>,
}

/// RFC 7009: responde 200 sempre (não revela se o token existia).
/// Revoga um refresh token (RFC 7009).
///
/// Responde 200 sempre, exista o token ou não.
#[utoipa::path(
    post, path = "/oauth/revoke", tag = "OAuth2",
    request_body(content = RevokeForm, content_type = "application/x-www-form-urlencoded"),
    responses(
        (status = 200, description = "Processado", body = Object),
        (status = 429, description = "Limite de taxa excedido", body = ErrorBody),
    )
)]
pub async fn revoke(State(st): State<AppState>, Form(f): Form<RevokeForm>) -> Result<Response, OAuthError> {
    if let (Some(token), Some(client_id)) = (f.token.as_deref(), f.client_id.as_deref())
        && token.len() == 43 && validate::valid_client_id(client_id) {
            store::revoke_by_token(&st.db, &crypto::hash_token(token), client_id).await?;
        }
    let mut resp = (StatusCode::OK, Json(json!({}))).into_response();
    no_store(resp.headers_mut());
    Ok(resp)
}

// ---------------------------------------------------------------- metadados

/// Metadados do servidor de autorização (RFC 8414).
#[utoipa::path(
    get, path = "/.well-known/oauth-authorization-server", tag = "OAuth2",
    responses((status = 200, description = "Metadados", body = AuthServerMetadata))
)]
pub async fn metadata(State(st): State<AppState>) -> Response {
    let iss = &st.cfg.issuer;
    Json(AuthServerMetadata {
        issuer: iss.clone(),
        authorization_endpoint: format!("{iss}/oauth/authorize"),
        token_endpoint: format!("{iss}/oauth/token"),
        revocation_endpoint: format!("{iss}/oauth/revoke"),
        response_types_supported: vec!["code".into()],
        grant_types_supported: vec!["authorization_code".into(), "refresh_token".into()],
        code_challenge_methods_supported: vec!["S256".into()],
        token_endpoint_auth_methods_supported: vec!["none".into()],
        scopes_supported: validate::SUPPORTED_SCOPES.iter().map(|s| s.to_string()).collect(),
        authorization_response_iss_parameter_supported: true,
    })
    .into_response()
}

/// Confere o código TOTP e o consome (cada código vale uma vez).
async fn check_totp(st: &AppState, user_id: uuid::Uuid, secret_enc: &[u8], code: &str) -> bool {
    let Some(secret) = st.mfa.decrypt(user_id, secret_enc) else {
        tracing::error!(%user_id, "segredo TOTP não decifra (PASSWORD_PEPPER trocado?)");
        return false;
    };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let Some(step) = crate::mfa::verify(&secret, code, crate::mfa::current_step(now)) else {
        return false;
    };
    store::consume_totp_step(&st.db, user_id, step).await.unwrap_or(false)
}

// ---------------------------------------------------------------- IP do cliente

/// Extrator do IP efetivo (par TCP ou `X-Forwarded-For` de proxy confiável).
pub struct ClientIp(pub std::net::IpAddr);

impl axum::extract::FromRequestParts<AppState> for ClientIp {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut axum::http::request::Parts, st: &AppState) -> Result<Self, Self::Rejection> {
        Ok(ClientIp(client_ip_from(&parts.extensions, &parts.headers, st.cfg.trust_proxy)))
    }
}
