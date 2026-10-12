//! Tipos de resposta/entrada das rotas. São a fonte ÚNICA do corpo real e do OpenAPI:
//! os handlers devolvem estes tipos, então a documentação não diverge do comportamento.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use utoipa::ToSchema;
use uuid::Uuid;

/// Erro das rotas de negócio.
#[derive(Serialize, ToSchema)]
pub struct ErrorBody {
    /// Código curto e estável (`invalid_token`, `rate_limited`, `payload_too_large`...).
    pub error: String,
    /// Mensagem legível; nunca contém detalhes internos.
    pub error_description: String,
}

/// Erro do endpoint de token (RFC 6749 §5.2).
#[derive(Serialize, ToSchema)]
pub struct OAuthErrorBody {
    /// `invalid_request`, `invalid_client`, `invalid_grant`, `unsupported_grant_type`, `invalid_scope`...
    pub error: String,
    pub error_description: String,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct MeResponse {
    pub id: Uuid,
    pub email: String,
    /// Escopos do token, separados por espaço.
    pub scope: String,
    pub client_id: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct RegisterResponse {
    pub id: Uuid,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OcrDocument {
    /// Texto extraído (corrido) ou uma linha de tabela no formato `Coluna: valor.`
    pub page_content: String,
    /// `source` (nome do arquivo enviado), `content_type` (`text` | `table_row`) e, para tabelas,
    /// `table_index`, `table_row_index` e `table_caption`.
    #[schema(value_type = Object)]
    pub metadata: Map<String, Value>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct OcrResponse {
    /// Nome do arquivo enviado, já sanitizado.
    pub filename: String,
    pub count: usize,
    pub documents: Vec<OcrDocument>,
}

/// Corpo multipart de `POST /v1/ocr`.
#[derive(ToSchema)]
pub struct OcrUpload {
    /// Único campo aceito. pdf, png, jpg, tif, bmp, webp, docx, pptx ou xlsx.
    #[schema(format = Binary, value_type = String)]
    pub file: Vec<u8>,
}

/// Formulário de login enviado pela página de `/oauth/authorize`.
#[derive(ToSchema)]
pub struct LoginForm {
    pub response_type: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: Option<String>,
    pub state: Option<String>,
    pub code_challenge: String,
    pub code_challenge_method: String,
    pub email: String,
    #[schema(format = Password)]
    pub password: String,
    /// Valor do campo oculto; deve ser igual ao cookie CSRF.
    pub csrf: String,
}

/// Metadados do servidor de autorização (RFC 8414).
#[derive(Serialize, ToSchema)]
pub struct AuthServerMetadata {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub revocation_endpoint: String,
    pub response_types_supported: Vec<String>,
    pub grant_types_supported: Vec<String>,
    pub code_challenge_methods_supported: Vec<String>,
    pub token_endpoint_auth_methods_supported: Vec<String>,
    pub scopes_supported: Vec<String>,
    pub authorization_response_iss_parameter_supported: bool,
}
