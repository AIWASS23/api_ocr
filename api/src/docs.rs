//! Documentação OpenAPI + Swagger UI (embutido no binário, sem CDN).
//!
//! Desligada por padrão: só existe com `ENABLE_DOCS=true`. A UI exige uma CSP um pouco mais
//! frouxa que a da API (estilos inline do React), por isso fica isolada nestas rotas.

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::http::{HeaderValue, header};
use tower_http::set_header::SetResponseHeaderLayer;
use utoipa::openapi::security::{AuthorizationCode, Flow, HttpAuthScheme, HttpBuilder, OAuth2, Scopes, SecurityScheme};
use utoipa::openapi::{Info, Server};
use utoipa::{Modify, OpenApi};
use utoipa_swagger_ui::{Config, SwaggerUi, oauth};

use crate::models::*;
use crate::oauth::TokenResponse;
use crate::state::AppState;

/// Cliente OAuth usado pelo botão "Authorize" do Swagger UI (registrado na partida quando as docs estão ligadas).
pub const DOCS_CLIENT_ID: &str = "swagger-ui";

#[derive(OpenApi)]
#[openapi(
    paths(
        crate::healthz_doc,
        crate::readyz_doc,
        crate::oauth::metadata,
        crate::oauth::authorize_get,
        crate::oauth::authorize_post,
        crate::oauth::token,
        crate::oauth::revoke,
        crate::users::register,
        crate::users::me,
        crate::ocr::ocr,
    ),
    components(schemas(
        ErrorBody, OAuthErrorBody, MeResponse, RegisterResponse, OcrDocument, OcrResponse, OcrUpload,
        LoginForm, AuthServerMetadata, TokenResponse,
        crate::oauth::TokenForm, crate::oauth::RevokeForm, crate::users::RegisterBody,
    )),
    tags(
        (name = "OCR", description = "Extração de texto e tabelas. Exige escopo `ocr`."),
        (name = "Conta", description = "Usuário autenticado e cadastro."),
        (name = "OAuth2", description = "Servidor de autorização: Authorization Code + PKCE (S256), refresh token com rotação."),
        (name = "Saúde", description = "Sondas para balanceador/orquestrador."),
    ),
    modifiers(&Dynamic),
)]
pub struct ApiDoc;

struct Dynamic;

impl Modify for Dynamic {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info = Info::builder()
            .title("OCR API")
            .version(env!("CARGO_PKG_VERSION"))
            .description(Some(
                "API de OCR protegida por OAuth2.\n\n\
                 **Como autenticar:** `Authorize` (Authorization Code + PKCE) ou envie `Authorization: Bearer <access_token>`.\n\n\
                 **Limites:** respostas `429` trazem `Retry-After`; `503` indica servidor ocupado ou sem memória.",
            ))
            .build();
    }
}

/// Documento com servidores e URLs OAuth do `ISSUER` real.
pub fn openapi(issuer: &str) -> utoipa::openapi::OpenApi {
    let mut api = ApiDoc::openapi();
    api.servers = Some(vec![Server::new(issuer)]);

    let flow = Flow::AuthorizationCode(AuthorizationCode::new(
        format!("{issuer}/oauth/authorize"),
        format!("{issuer}/oauth/token"),
        Scopes::from_iter([("ocr", "Usar o OCR"), ("profile", "Ler o próprio perfil")]),
    ));
    let components = api.components.get_or_insert_with(Default::default);
    components.add_security_scheme("oauth2", SecurityScheme::OAuth2(OAuth2::new([flow])));
    components.add_security_scheme(
        "bearer",
        SecurityScheme::Http(HttpBuilder::new().scheme(HttpAuthScheme::Bearer).bearer_format("JWT").build()),
    );
    api
}

/// CSP da UI: scripts só do próprio servidor; estilos inline (exigidos pelo Swagger UI).
const DOCS_CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; \
    img-src 'self' data:; font-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; \
    base-uri 'none'; form-action 'self'";

pub fn router(st: &AppState) -> Router<AppState> {
    let issuer = &st.cfg.issuer;
    let ui_config = Config::new(["/openapi.json"])
        .oauth2_redirect_url(format!("{issuer}/docs/oauth2-redirect.html"))
        .persist_authorization(false)
        .try_it_out_enabled(true);
    let oauth_config = oauth::Config::new()
        .client_id(DOCS_CLIENT_ID)
        .scopes(vec!["ocr".to_string(), "profile".to_string()])
        .use_pkce_with_authorization_code_grant(true);

    let swagger = SwaggerUi::new("/docs")
        .url("/openapi.json", openapi(issuer))
        .config(ui_config)
        .oauth(oauth_config);

    Router::<AppState>::from(swagger)
        .layer(DefaultBodyLimit::max(1024))
        .layer(SetResponseHeaderLayer::overriding(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(DOCS_CSP),
        ))
}

/// Registra (ou atualiza) o cliente OAuth do Swagger UI: redirect só para a própria origem.
pub async fn ensure_client(db: &sqlx::PgPool, issuer: &str) -> Result<(), sqlx::Error> {
    crate::store::ensure_client(
        db,
        DOCS_CLIENT_ID,
        "Swagger UI",
        &format!("{issuer}/docs/oauth2-redirect.html"),
        &["ocr".to_string(), "profile".to_string()],
    )
    .await
}
