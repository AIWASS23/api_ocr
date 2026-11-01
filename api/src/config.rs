use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone)]
pub struct Config {
    pub database_url: String,
    pub bind_addr: SocketAddr,
    /// URL pública (sem barra final). Vira `iss`/`aud` dos tokens.
    pub issuer: String,
    pub jwt_secret: Vec<u8>,
    /// Segredos JWT anteriores: ainda VALIDAM tokens (durante a rotação), nunca assinam.
    pub jwt_previous_secrets: Vec<Vec<u8>>,
    pub password_pepper: Vec<u8>,
    /// Confia no último valor de `X-Forwarded-For` (use só atrás de UM proxy confiável).
    pub trust_proxy: bool,
    pub allow_registration: bool,
    /// Serve /docs (Swagger UI) e /openapi.json. Desligado por padrão.
    pub enable_docs: bool,
    pub cors_origins: Vec<String>,

    pub access_token_ttl: Duration,
    pub refresh_token_ttl: Duration,
    /// Vida máxima absoluta de uma cadeia de refresh tokens.
    pub refresh_family_max: Duration,
    pub auth_code_ttl: Duration,

    // OCR
    pub ocr_max_upload_bytes: usize,
    pub ocr_max_pages: usize,
    pub ocr_max_concurrent: usize,
    pub ocr_timeout: Duration,
    /// Não aceita novo OCR com uso de memória acima disto (% do limite do cgroup/host). 100 desliga.
    pub ocr_memory_watermark_pct: u8,
    /// Binário executado como worker de OCR (o próprio executável, por padrão).
    pub ocr_worker_exe: PathBuf,
    /// Teto opcional de espaço de endereçamento de cada worker, em MiB. 0 (padrão) = sem teto: a
    /// memória real é limitada pelo cgroup, e o worker é quem morre se ela acabar.
    pub ocr_worker_memory_mb: u64,
    /// Teto de RSS por worker, em MiB, imposto pelo vigia da API (SIGKILL ao passar). 0 = só o
    /// limite do contêiner (o vigia mata o maior worker ao chegar a 90% dele).
    pub ocr_worker_max_rss_mb: u64,

    /// Chunking (`/v1/chunk`).
    pub chunk: ChunkSettings,
    /// Embeddings (`/v1/embedding`).
    pub embedding: EmbeddingSettings,

    // Limites de conexão / taxa
    pub max_connections: usize,
    pub max_connections_per_ip: usize,
    pub max_in_flight: usize,
    pub rate_general_per_min: u32,
    pub rate_auth_per_min: u32,
    pub rate_ocr_per_min: u32,
    pub rate_chunk_per_min: u32,
    pub rate_embedding_per_min: u32,
    /// Redis (redis:// ou rediss://) para o limite de taxa global entre réplicas. Opcional.
    pub rate_limit_redis_url: Option<String>,
    pub rate_limit_redis_prefix: String,

    /// Falhas de login vindas de um mesmo IP, em 5 minutos, que disparam um alerta no log.
    pub security_alert_login_failures: u32,
}

/// Limites e modelos do chunking. Os padrões valem para a API e para os testes.
#[derive(Clone, Debug)]
pub struct ChunkSettings {
    /// Maior texto aceito por requisição (soma de todos os documentos), em bytes de UTF-8.
    pub max_text_bytes: usize,
    pub max_documents: usize,
    /// Mais chunks que uma requisição pode gerar (a resposta inteira trafega de uma vez).
    pub max_chunks: usize,
    /// Chunkings simultâneos no servidor (cada um é um processo worker).
    pub max_concurrent: usize,
    /// Prazo de cada chunking (o worker é morto ao estourar). Texto legítimo leva poucos
    /// segundos; HTML/JSON patológico (aninhamento profundo deixa o parser HTML quadrático)
    /// gasta no máximo isto de CPU de um worker.
    pub timeout: Duration,
    /// Teto de RSS por worker de chunking, em MiB (0 = só o limite do contêiner).
    pub worker_max_rss_mb: u64,
    /// Tokenizador Hugging Face LOCAL (diretório com `tokenizer.json`, ou o próprio arquivo) das
    /// estratégias medidas em tokens. Sem ele, essas estratégias aparecem como indisponíveis. O
    /// worker não tem rede: um id do Hub não funcionaria.
    pub tokenizer: Option<PathBuf>,
    /// Diretório de dados do NLTK com `tokenizers/punkt_tab` (além de `/usr/share/nltk_data` e
    /// afins, que já são lidos). Sem o `punkt_tab`, a estratégia `nltk` usa `sentence`.
    pub nltk_data: Option<PathBuf>,
}

impl Default for ChunkSettings {
    fn default() -> Self {
        Self {
            max_text_bytes: 5 * 1024 * 1024,
            max_documents: 10_000,
            max_chunks: 20_000,
            max_concurrent: 4,
            timeout: Duration::from_secs(30),
            worker_max_rss_mb: 1024,
            tokenizer: None,
            nltk_data: None,
        }
    }
}

impl ChunkSettings {
    fn from_env() -> Result<Self, ConfigError> {
        let d = Self::default();
        let s = Self {
            max_text_bytes: parse::<usize>("CHUNK_MAX_TEXT_MB", d.max_text_bytes / (1024 * 1024))? * 1024 * 1024,
            max_documents: parse("CHUNK_MAX_DOCUMENTS", d.max_documents)?,
            max_chunks: parse("CHUNK_MAX_CHUNKS", d.max_chunks)?,
            max_concurrent: parse("CHUNK_MAX_CONCURRENT", d.max_concurrent)?,
            timeout: Duration::from_secs(parse("CHUNK_TIMEOUT_SECS", d.timeout.as_secs())?),
            worker_max_rss_mb: parse("CHUNK_WORKER_MAX_RSS_MB", d.worker_max_rss_mb)?,
            tokenizer: var("CHUNK_TOKENIZER").map(PathBuf::from),
            nltk_data: var("CHUNK_NLTK_DATA").map(PathBuf::from),
        };
        // Configurado e inexistente = erro de configuração: falha na partida, não na 1ª requisição.
        if let Some(t) = &s.tokenizer
            && !crate::chunk::tokenizer_json(t).is_file()
        {
            return Err(ConfigError("CHUNK_TOKENIZER precisa apontar para um tokenizer.json local (ou o diretório dele)".into()));
        }
        if let Some(d) = &s.nltk_data
            && !d.is_dir()
        {
            return Err(ConfigError("CHUNK_NLTK_DATA precisa ser um diretório".into()));
        }
        Ok(s)
    }
}

/// Limites e modelos dos embeddings. O worker não tem rede: só modelos LOCAIS.
#[derive(Clone, Debug)]
pub struct EmbeddingSettings {
    /// Textos (ou documentos) por requisição; vale também para `queries`.
    pub max_texts: usize,
    /// Soma dos textos e consultas, em bytes de UTF-8.
    pub max_text_bytes: usize,
    pub max_images: usize,
    /// Soma das imagens decodificadas, em bytes.
    pub max_image_bytes: usize,
    pub max_concurrent: usize,
    /// Prazo de cada pedido (carga do modelo + cálculo); o worker é morto ao estourar.
    pub timeout: Duration,
    /// Teto de RSS por worker, em MiB (0 = só o limite do contêiner).
    pub worker_max_rss_mb: u64,
    pub batch_size: usize,
    /// `cpu`, ou `cuda`/`auto` com a feature `cuda`.
    pub device: String,
    /// Diretório local do modelo de cada estratégia (`EMBEDDING_MODELS=e5=/models/e5,...`).
    pub models: HashMap<String, PathBuf>,
}

impl Default for EmbeddingSettings {
    fn default() -> Self {
        Self {
            max_texts: 256,
            max_text_bytes: 2 * 1024 * 1024,
            max_images: 16,
            max_image_bytes: 20 * 1024 * 1024,
            max_concurrent: 2,
            timeout: Duration::from_secs(120),
            worker_max_rss_mb: 0,
            batch_size: 32,
            device: "cpu".into(),
            models: HashMap::new(),
        }
    }
}

/// `estrategia=/caminho,...`: estratégia com modelo e diretório local com `tokenizer.json`.
pub fn parse_embedding_models(raw: &str) -> Result<HashMap<String, PathBuf>, ConfigError> {
    let mut out = HashMap::new();
    for item in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (k, v) = item
            .split_once('=')
            .ok_or_else(|| ConfigError(format!("EMBEDDING_MODELS: '{item}' deve ser estrategia=/caminho")))?;
        let strategy = embedding::normalize_strategy(k.trim()).map_err(|e| ConfigError(format!("EMBEDDING_MODELS: {e}")))?;
        if !embedding::MODEL_STRATEGIES.contains(&strategy) || strategy == "ollama" {
            return Err(ConfigError(format!("EMBEDDING_MODELS: a estratégia '{strategy}' não usa diretório de modelo")));
        }
        let dir = PathBuf::from(v.trim());
        if !dir.join("tokenizer.json").is_file() {
            return Err(ConfigError(format!("EMBEDDING_MODELS: {} não é um diretório de modelo (sem tokenizer.json)", dir.display())));
        }
        out.insert(strategy.to_string(), dir);
    }
    Ok(out)
}

impl EmbeddingSettings {
    fn from_env() -> Result<Self, ConfigError> {
        let d = Self::default();
        let mb = 1024 * 1024;
        let s = Self {
            max_texts: parse("EMBEDDING_MAX_TEXTS", d.max_texts)?,
            max_text_bytes: parse::<usize>("EMBEDDING_MAX_TEXT_MB", d.max_text_bytes / mb)? * mb,
            max_images: parse("EMBEDDING_MAX_IMAGES", d.max_images)?,
            max_image_bytes: parse::<usize>("EMBEDDING_MAX_IMAGE_MB", d.max_image_bytes / mb)? * mb,
            max_concurrent: parse("EMBEDDING_MAX_CONCURRENT", d.max_concurrent)?,
            timeout: Duration::from_secs(parse("EMBEDDING_TIMEOUT_SECS", d.timeout.as_secs())?),
            worker_max_rss_mb: parse("EMBEDDING_WORKER_MAX_RSS_MB", d.worker_max_rss_mb)?,
            batch_size: parse("EMBEDDING_BATCH_SIZE", d.batch_size)?,
            device: var("EMBEDDING_DEVICE").unwrap_or(d.device),
            models: var("EMBEDDING_MODELS").map_or_else(|| Ok(HashMap::new()), |v| parse_embedding_models(&v))?,
        };
        embedding::core::onnx::check_device(&s.device).map_err(|e| ConfigError(format!("EMBEDDING_DEVICE: {e}")))?;
        Ok(s)
    }
}

// Nunca imprime segredos.
impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("bind_addr", &self.bind_addr)
            .field("issuer", &self.issuer)
            .field("trust_proxy", &self.trust_proxy)
            .field("allow_registration", &self.allow_registration)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("configuração inválida: {0}")]
pub struct ConfigError(String);

/// Fora de loopback/socket local, exige TLS com o banco (`sslmode=require`, `verify-ca` ou
/// `verify-full`). Uma rede privada e isolada pode abrir mão com `DATABASE_ALLOW_PLAINTEXT=true`.
pub fn check_database_tls(url: &str, allow_plaintext: bool) -> Result<(), ConfigError> {
    use sqlx::postgres::{PgConnectOptions, PgSslMode};
    let opts: PgConnectOptions = url.parse().map_err(|_| ConfigError("DATABASE_URL inválida".into()))?;
    let local = opts.get_socket().is_some()
        || opts.get_host().starts_with('/')
        || matches!(opts.get_host(), "localhost" | "127.0.0.1" | "::1" | "[::1]");
    let tls = matches!(opts.get_ssl_mode(), PgSslMode::Require | PgSslMode::VerifyCa | PgSslMode::VerifyFull);
    if local || tls || allow_plaintext {
        return Ok(());
    }
    Err(ConfigError(
        "DATABASE_URL aponta para um host remoto sem TLS: acrescente ?sslmode=verify-full (ou require); \
         numa rede privada isolada, DATABASE_ALLOW_PLAINTEXT=true"
            .into(),
    ))
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn required(name: &str) -> Result<String, ConfigError> {
    var(name).ok_or_else(|| ConfigError(format!("{name} não definida")))
}

fn parse<T: std::str::FromStr>(name: &str, default: T) -> Result<T, ConfigError> {
    match var(name) {
        None => Ok(default),
        Some(v) => v
            .trim()
            .parse()
            .map_err(|_| ConfigError(format!("{name} com valor inválido"))),
    }
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let jwt_secret = required("JWT_SECRET")?.into_bytes();
        let password_pepper = required("PASSWORD_PEPPER")?.into_bytes();
        if jwt_secret.len() < 32 {
            return Err(ConfigError("JWT_SECRET precisa de ao menos 32 caracteres".into()));
        }
        if password_pepper.len() < 32 {
            return Err(ConfigError("PASSWORD_PEPPER precisa de ao menos 32 caracteres".into()));
        }
        if jwt_secret == password_pepper {
            return Err(ConfigError("JWT_SECRET e PASSWORD_PEPPER devem ser diferentes".into()));
        }
        // Rotação: o segredo antigo continua validando tokens até eles expirarem.
        let jwt_previous_secrets: Vec<Vec<u8>> = var("JWT_SECRET_PREVIOUS")
            .map(|v| v.split(',').map(|s| s.trim().as_bytes().to_vec()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default();
        for old in &jwt_previous_secrets {
            if old.len() < 32 {
                return Err(ConfigError("JWT_SECRET_PREVIOUS: cada segredo precisa de ao menos 32 caracteres".into()));
            }
            if old == &password_pepper {
                return Err(ConfigError("JWT_SECRET_PREVIOUS não pode ser igual ao PASSWORD_PEPPER".into()));
            }
        }

        let database_url = required("DATABASE_URL")?;
        check_database_tls(&database_url, parse("DATABASE_ALLOW_PLAINTEXT", false)?)?;

        let issuer = required("ISSUER")?.trim_end_matches('/').to_string();
        let issuer_url = url::Url::parse(&issuer)
            .map_err(|_| ConfigError("ISSUER não é uma URL válida".into()))?;
        let local = matches!(issuer_url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
        if issuer_url.scheme() != "https" && !local {
            return Err(ConfigError("ISSUER precisa ser https (exceto localhost)".into()));
        }

        let cfg = Self {
            database_url,
            bind_addr: parse("BIND_ADDR", "127.0.0.1:8080".parse().expect("literal"))?,
            issuer,
            jwt_secret,
            jwt_previous_secrets,
            password_pepper,
            trust_proxy: parse("TRUST_PROXY", false)?,
            allow_registration: parse("ALLOW_REGISTRATION", false)?,
            enable_docs: parse("ENABLE_DOCS", false)?,
            cors_origins: var("CORS_ORIGINS")
                .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
                .unwrap_or_default(),

            access_token_ttl: Duration::from_secs(parse("ACCESS_TOKEN_TTL_SECS", 600)?),
            refresh_token_ttl: Duration::from_secs(parse("REFRESH_TOKEN_TTL_SECS", 30 * 86_400)?),
            refresh_family_max: Duration::from_secs(parse("REFRESH_FAMILY_MAX_SECS", 90 * 86_400)?),
            auth_code_ttl: Duration::from_secs(60),

            ocr_max_upload_bytes: parse::<usize>("OCR_MAX_UPLOAD_MB", 25)? * 1024 * 1024,
            ocr_max_pages: parse("OCR_MAX_PAGES", 100)?,
            ocr_max_concurrent: parse("OCR_MAX_CONCURRENT", 2)?,
            ocr_timeout: Duration::from_secs(parse("OCR_TIMEOUT_SECS", 180)?),
            ocr_memory_watermark_pct: parse("OCR_MEMORY_WATERMARK_PCT", 85)?,
            ocr_worker_exe: match var("OCR_WORKER_EXE") {
                Some(p) => PathBuf::from(p),
                None => std::env::current_exe().map_err(|e| ConfigError(format!("executável atual: {e}")))?,
            },
            ocr_worker_memory_mb: parse("OCR_WORKER_MEMORY_MB", 0)?,
            ocr_worker_max_rss_mb: parse("OCR_WORKER_MAX_RSS_MB", 0)?,
            chunk: ChunkSettings::from_env()?,
            embedding: EmbeddingSettings::from_env()?,

            max_connections: parse("MAX_CONNECTIONS", 1024)?,
            max_connections_per_ip: parse("MAX_CONNECTIONS_PER_IP", 64)?,
            max_in_flight: parse("MAX_IN_FLIGHT", 256)?,
            rate_general_per_min: parse("RATE_GENERAL_PER_MIN", 300)?,
            rate_auth_per_min: parse("RATE_AUTH_PER_MIN", 20)?,
            rate_ocr_per_min: parse("RATE_OCR_PER_MIN", 10)?,
            rate_chunk_per_min: parse("RATE_CHUNK_PER_MIN", 60)?,
            rate_embedding_per_min: parse("RATE_EMBEDDING_PER_MIN", 30)?,
            rate_limit_redis_url: var("RATE_LIMIT_REDIS_URL"),
            rate_limit_redis_prefix: var("RATE_LIMIT_REDIS_PREFIX").unwrap_or_else(|| "api_ocr:".to_string()),
            security_alert_login_failures: parse("SECURITY_ALERT_LOGIN_FAILURES", 20)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.access_token_ttl > Duration::from_secs(3600) {
            return Err(ConfigError("ACCESS_TOKEN_TTL_SECS não pode passar de 3600".into()));
        }
        for (name, v) in [
            ("OCR_MAX_CONCURRENT", self.ocr_max_concurrent),
            ("OCR_MAX_PAGES", self.ocr_max_pages),
            ("MAX_CONNECTIONS", self.max_connections),
            ("MAX_CONNECTIONS_PER_IP", self.max_connections_per_ip),
            ("MAX_IN_FLIGHT", self.max_in_flight),
            ("CHUNK_MAX_TEXT_MB", self.chunk.max_text_bytes),
            ("CHUNK_MAX_DOCUMENTS", self.chunk.max_documents),
            ("CHUNK_MAX_CHUNKS", self.chunk.max_chunks),
            ("CHUNK_MAX_CONCURRENT", self.chunk.max_concurrent),
            ("EMBEDDING_MAX_TEXTS", self.embedding.max_texts),
            ("EMBEDDING_MAX_TEXT_MB", self.embedding.max_text_bytes),
            ("EMBEDDING_MAX_CONCURRENT", self.embedding.max_concurrent),
            ("EMBEDDING_BATCH_SIZE", self.embedding.batch_size),
        ] {
            if v == 0 {
                return Err(ConfigError(format!("{name} precisa ser maior que 0")));
            }
        }
        if self.embedding.timeout.is_zero() {
            return Err(ConfigError("EMBEDDING_TIMEOUT_SECS precisa ser maior que 0".into()));
        }
        if self.chunk.timeout.is_zero() {
            return Err(ConfigError("CHUNK_TIMEOUT_SECS precisa ser maior que 0".into()));
        }
        if !(1..=100).contains(&self.ocr_memory_watermark_pct) {
            return Err(ConfigError("OCR_MEMORY_WATERMARK_PCT precisa estar entre 1 e 100".into()));
        }
        for (name, v) in [
            ("RATE_GENERAL_PER_MIN", self.rate_general_per_min),
            ("RATE_AUTH_PER_MIN", self.rate_auth_per_min),
            ("RATE_OCR_PER_MIN", self.rate_ocr_per_min),
            ("RATE_CHUNK_PER_MIN", self.rate_chunk_per_min),
            ("RATE_EMBEDDING_PER_MIN", self.rate_embedding_per_min),
            ("SECURITY_ALERT_LOGIN_FAILURES", self.security_alert_login_failures),
        ] {
            if v == 0 {
                return Err(ConfigError(format!("{name} precisa ser maior que 0")));
            }
        }
        Ok(())
    }

    /// Todas as chaves que validam tokens: a atual primeiro.
    pub fn jwt_verification_secrets(&self) -> Vec<&[u8]> {
        std::iter::once(self.jwt_secret.as_slice()).chain(self.jwt_previous_secrets.iter().map(Vec::as_slice)).collect()
    }

    /// Cookies `__Host-` exigem HTTPS.
    pub fn secure_cookies(&self) -> bool {
        self.issuer.starts_with("https://")
    }
}

#[cfg(test)]
mod tests {
    use super::check_database_tls;

    #[test]
    fn remote_database_requires_tls() {
        assert!(check_database_tls("postgres://u:p@127.0.0.1:5432/db", false).is_ok());
        assert!(check_database_tls("postgres://u:p@localhost/db", false).is_ok());
        assert!(check_database_tls("postgres://u:p@db.exemplo.com/db?sslmode=verify-full", false).is_ok());
        assert!(check_database_tls("postgres://u:p@db.exemplo.com/db?sslmode=require", false).is_ok());
        assert!(check_database_tls("postgres://u:p@db.exemplo.com/db", false).is_err());
        assert!(check_database_tls("postgres://u:p@db.exemplo.com/db?sslmode=prefer", false).is_err());
        assert!(check_database_tls("postgres://u:p@db.exemplo.com/db?sslmode=disable", false).is_err());
        assert!(check_database_tls("postgres://u:p@db/db", true).is_ok(), "opt-out explícito");
        assert!(check_database_tls("postgres:///db?host=/run/postgresql", false).is_ok(), "socket UNIX");
    }
}
