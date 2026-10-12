use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use uuid::Uuid;

use crate::error::ApiError;
use crate::state::AppState;
use crate::tokens::Claims;

/// Extrator das rotas protegidas: exige `Authorization: Bearer <JWT válido>`.
pub struct Auth(pub Claims);

impl Auth {
    pub fn user_id(&self) -> Result<Uuid, ApiError> {
        Uuid::parse_str(&self.0.sub).map_err(|_| ApiError::Unauthorized)
    }

    pub fn require_scope(&self, scope: &'static str) -> Result<(), ApiError> {
        if self.0.has_scope(scope) {
            Ok(())
        } else {
            Err(ApiError::Forbidden("escopo insuficiente para esta operação"))
        }
    }
}

impl FromRequestParts<AppState> for Auth {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, ApiError> {
        let mut values = parts.headers.get_all(AUTHORIZATION).iter();
        let (Some(value), None) = (values.next(), values.next()) else {
            return Err(ApiError::Unauthorized);
        };
        let value = value.to_str().map_err(|_| ApiError::Unauthorized)?;
        if value.len() > 4096 {
            return Err(ApiError::Unauthorized);
        }
        let (scheme, token) = value.split_once(' ').ok_or(ApiError::Unauthorized)?;
        if !scheme.eq_ignore_ascii_case("bearer") || token.is_empty() || token.contains(' ') {
            return Err(ApiError::Unauthorized);
        }
        let claims = state.tokens.verify(token).ok_or(ApiError::Unauthorized)?;

        // Revogação imediata: usuário desativado, ou sessões derrubadas depois da emissão
        // (reuso de refresh token, `api_ocr revoke-sessions`). Uma consulta por chave primária.
        let user = Uuid::parse_str(&claims.sub).map_err(|_| ApiError::Unauthorized)?;
        let gate = crate::store::token_gate(&state.db, user).await?.ok_or(ApiError::Unauthorized)?;
        if !gate.is_active || gate.valid_after.is_some_and(|cut| claims.iat as i64 <= cut) {
            return Err(ApiError::Unauthorized);
        }
        Ok(Auth(claims))
    }
}
