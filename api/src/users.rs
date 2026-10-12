use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Deserialize;
use uuid::Uuid;

use crate::auth::Auth;
use crate::crypto::HashError;
use crate::error::ApiError;
use crate::models::{ErrorBody, MeResponse, RegisterResponse};
use crate::oauth::ClientIp;
use crate::state::AppState;
use crate::{store, validate};

#[derive(Deserialize, utoipa::ToSchema)]
#[serde(deny_unknown_fields)]
pub struct RegisterBody {
    /// E-mail (normalizado para minúsculas).
    #[schema(example = "voce@exemplo.com")]
    email: String,
    /// 12 a 128 caracteres.
    #[schema(format = Password, min_length = 12, max_length = 128)]
    password: String,
}

/// Cadastra um usuário.
///
/// Só existe com `ALLOW_REGISTRATION=true`; desligado, responde 404. Limitado por IP.
#[utoipa::path(
    post, path = "/auth/register", tag = "Conta",
    request_body = RegisterBody,
    responses(
        (status = 201, description = "Usuário criado", body = RegisterResponse),
        (status = 400, description = "E-mail ou senha inválidos", body = ErrorBody),
        (status = 404, description = "Cadastro público desligado", body = ErrorBody),
        (status = 409, description = "Não foi possível concluir o cadastro", body = ErrorBody),
        (status = 413, description = "Corpo grande demais", body = ErrorBody),
        (status = 429, description = "Limite de taxa excedido", body = ErrorBody),
    )
)]
pub async fn register(State(st): State<AppState>, ip: ClientIp, Json(b): Json<RegisterBody>) -> Result<(StatusCode, Json<RegisterResponse>), ApiError> {
    if !st.cfg.allow_registration {
        return Err(ApiError::NotFound);
    }
    let email = validate::normalize_email(&b.email).ok_or(ApiError::BadRequest("e-mail inválido"))?;
    validate::check_password(&b.password, &email).map_err(ApiError::BadRequest)?;

    let hash = st.passwords.hash(b.password).await.map_err(|e| match e {
        HashError::Busy => ApiError::Unavailable(std::time::Duration::from_secs(3)),
        HashError::Internal => ApiError::internal("hash de senha", "falha"),
    })?;

    let id = Uuid::new_v4();
    if !store::create_user(&st.db, id, &email, &hash).await? {
        return Err(ApiError::Conflict("não foi possível concluir o cadastro"));
    }
    crate::security::event(&st, "user_registered", Some(id), None, Some(&ip.0.to_string()), None).await;
    Ok((StatusCode::CREATED, Json(RegisterResponse { id })))
}

/// Dados do usuário autenticado.
#[utoipa::path(
    get, path = "/v1/me", tag = "Conta",
    security(("bearer" = []), ("oauth2" = ["profile"])),
    responses(
        (status = 200, description = "Usuário do token", body = MeResponse),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 429, description = "Limite de taxa excedido", body = ErrorBody),
    )
)]
pub async fn me(State(st): State<AppState>, auth: Auth) -> Result<Json<MeResponse>, ApiError> {
    let id = auth.user_id()?;
    let (email, created_at) = store::find_email_by_id(&st.db, id).await?.ok_or(ApiError::Unauthorized)?;
    Ok(Json(MeResponse { id, email, scope: auth.0.scope, client_id: auth.0.client_id, created_at }))
}
