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

/// Corpo multipart de `POST /v1/data_preparation`.
#[derive(ToSchema)]
pub struct OcrUpload {
    /// Único campo aceito. pdf, png, jpg, tif, bmp, webp, docx, pptx ou xlsx.
    #[schema(format = Binary, value_type = String)]
    pub file: Vec<u8>,
}

/// Documento de entrada do chunking (o mesmo formato de `OcrDocument`).
#[derive(Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct InputDocument {
    pub page_content: String,
    /// Copiados para cada chunk deste documento (até 4 KB em JSON).
    #[serde(default)]
    #[schema(value_type = Object)]
    pub metadata: Map<String, Value>,
}

/// Corpo de `POST /v1/chunk`. Envie `text` OU `documents`.
#[derive(Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ChunkRequest {
    /// Estratégia: nome ou apelido (ver `GET /v1/chunk/strategies`).
    #[schema(example = "recursive")]
    pub strategy: String,
    /// Tamanho máximo de cada chunk, na unidade da estratégia (caracteres ou tokens).
    /// Padrão: o da estratégia.
    #[schema(example = 1000)]
    pub chunk_size: Option<usize>,
    /// Sobreposição entre chunks vizinhos, só nas estratégias que a usam. Padrão: 10% do
    /// `chunk_size`; máximo: metade dele.
    #[schema(example = 100)]
    pub chunk_overlap: Option<usize>,
    /// Texto puro.
    pub text: Option<String>,
    /// Documentos já extraídos, por exemplo a lista `documents` de `POST /v1/data_preparation`.
    pub documents: Option<Vec<InputDocument>>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChunkOut {
    pub page_content: String,
    /// Metadados do documento de origem somados aos da estratégia (`h1`/`h2`, `breadcrumb`,
    /// `artigo`, `parent_id`, `parent_content`, `question`...).
    #[schema(value_type = Object)]
    pub metadata: Map<String, Value>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChunkResponse {
    /// Nome canônico da estratégia usada.
    pub strategy: String,
    /// Unidade do `chunk_size`: `characters` ou `tokens`.
    pub unit: String,
    pub chunk_size: usize,
    /// `null` quando a estratégia não usa sobreposição.
    pub chunk_overlap: Option<usize>,
    pub count: usize,
    pub chunks: Vec<ChunkOut>,
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct SizeParam {
    /// `characters` ou `tokens`.
    pub unit: String,
    pub default: usize,
    pub min: usize,
    pub max: usize,
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct OverlapParam {
    /// 10% do `chunk_size` padrão (se o `chunk_size` for informado, 10% dele).
    pub default: usize,
    pub min: usize,
    /// O máximo é esta fração do `chunk_size`.
    pub max_fraction_of_chunk_size: f64,
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct StrategyInfo {
    /// Nome canônico.
    pub name: String,
    /// Outros nomes aceitos em `strategy`.
    pub aliases: Vec<String>,
    pub description: String,
    /// Se pode ser usada neste servidor.
    pub available: bool,
    /// Por que não está disponível (só quando `available` é `false`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    /// Observação sobre o comportamento neste servidor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub chunk_size: SizeParam,
    /// `null` quando a estratégia não usa sobreposição.
    pub chunk_overlap: Option<OverlapParam>,
}

/// Limites de `POST /v1/chunk` neste servidor.
#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct ChunkLimits {
    /// Texto máximo por requisição (soma de todos os documentos), em bytes UTF-8.
    pub max_text_bytes: usize,
    pub max_documents: usize,
    /// Chunks máximos gerados por requisição.
    pub max_chunks: usize,
    /// `metadata` máximo de cada documento de entrada, em bytes de JSON.
    pub max_metadata_bytes: usize,
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct StrategiesResponse {
    pub count: usize,
    pub strategies: Vec<StrategyInfo>,
    pub limits: ChunkLimits,
}

/// Corpo de `POST /v1/embedding`. Envie UM de `texts`, `documents` ou `images`.
#[derive(Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingRequest {
    /// Estratégia: nome ou apelido (ver `GET /v1/embedding/strategies`).
    #[schema(example = "e5")]
    pub strategy: String,
    /// `document` (padrão) ou `query`: modo de documento ou de consulta do modelo.
    pub input_type: Option<String>,
    pub texts: Option<Vec<String>>,
    /// Documentos (ex.: a lista `chunks` de `POST /v1/chunk`); só o `page_content` é usado.
    pub documents: Option<Vec<InputDocument>>,
    /// Imagens em base64 (PNG, JPEG, WebP, BMP, TIFF, GIF); só em `clip`/`siglip`.
    pub images: Option<Vec<String>>,
    /// Consultas, só em `tfidf`/`bm25`/`lsa` (vetorizadas com o vocabulário dos documentos).
    pub queries: Option<Vec<String>>,
    /// Corta os vetores (Matryoshka); nas esparsas, tamanho do vocabulário/hash.
    pub dimensions: Option<usize>,
    /// Vetores com norma 1. Padrão: `false`.
    pub normalize: Option<bool>,
    /// `dense` ou `sparse` (padrão: o da estratégia).
    pub format: Option<String>,
}

/// Vetor esparso: posições não nulas e seus valores.
#[derive(Serialize, Deserialize, ToSchema)]
pub struct SparseVectorOut {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct EmbeddingResponse {
    /// Nome canônico da estratégia usada.
    pub strategy: String,
    /// Modelo configurado no servidor (ou o nome da estratégia, quando ela não usa modelo).
    pub model: String,
    pub input_type: String,
    /// `dense` ou `sparse`.
    pub format: String,
    /// Tamanho dos vetores (nas esparsas, do vocabulário/hash).
    pub dimension: usize,
    pub count: usize,
    /// Um vetor por entrada (formato `dense`).
    pub embeddings: Option<Vec<Vec<f32>>>,
    /// Um vetor esparso por entrada (formato `sparse`).
    pub sparse_embeddings: Option<Vec<SparseVectorOut>>,
    /// Vetores de `queries` (`tfidf`, `bm25`, `lsa`).
    pub query_embeddings: Option<Vec<Vec<f32>>>,
    pub sparse_query_embeddings: Option<Vec<SparseVectorOut>>,
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingStrategyInfo {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
    /// `dense`, `sparse` ou `multimodal`.
    pub kind: String,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
    /// Modelo configurado neste servidor (sem o caminho).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// `dense` ou `sparse`.
    pub default_format: String,
    pub accepts_images: bool,
    /// Aprende o vocabulário com os documentos da própria requisição (envie `queries` junto).
    pub fitted_per_request: bool,
}

/// Limites de `POST /v1/embedding` neste servidor.
#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingLimits {
    pub max_texts: usize,
    /// Soma de textos e consultas, em bytes UTF-8.
    pub max_text_bytes: usize,
    pub max_images: usize,
    /// Soma das imagens decodificadas, em bytes.
    pub max_image_bytes: usize,
    pub max_dimensions: usize,
}

#[derive(Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingStrategiesResponse {
    pub count: usize,
    pub strategies: Vec<EmbeddingStrategyInfo>,
    pub limits: EmbeddingLimits,
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
