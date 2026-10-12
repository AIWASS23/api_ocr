//! Acesso ao banco. REGRA: nenhum SQL é montado por concatenação/format!;
//! todo valor externo entra por `.bind(...)` (consultas parametrizadas).

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Debug, sqlx::FromRow)]
pub struct UserRow {
    pub id: Uuid,
    pub password_hash: String,
    pub is_active: bool,
    pub totp_secret_enc: Option<Vec<u8>>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ClientRow {
    pub client_id: String,
    pub name: String,
    pub redirect_uris: Vec<String>,
    pub allowed_scopes: Vec<String>,
    pub is_active: bool,
}

#[derive(Debug, sqlx::FromRow)]
pub struct AuthCodeRow {
    pub client_id: String,
    pub user_id: Uuid,
    pub redirect_uri: String,
    pub scope: String,
    pub code_challenge: String,
    pub expired: bool,
}

pub enum CodeOutcome {
    Valid(AuthCodeRow),
    /// O código já tinha sido trocado: possível roubo.
    Reused { user_id: Uuid, client_id: String },
    Unknown,
}

pub enum RotateOutcome {
    Rotated { user_id: Uuid, scope: String },
    InvalidScope,
    /// Token desconhecido, expirado ou revogado.
    Invalid,
    /// Token já usado: a família inteira foi revogada.
    Reuse { user_id: Uuid },
}

/// Falhas seguidas de UMA origem (IP / prefixo /64) que bloqueiam a conta só para ela.
pub const MAX_FAILED_LOGINS: i32 = 5;
/// Falhas de TODAS as origens (em 24 h) que bloqueiam a conta para origens desconhecidas.
pub const ACCOUNT_MAX_FAILED_LOGINS: i32 = 50;
const LOCK_BASE_SECS: f64 = 60.0;
/// Origem onde o usuário entrou com sucesso nos últimos N dias ignora o bloqueio da conta inteira.
const KNOWN_ORIGIN_DAYS: f64 = 90.0;

// ---------------------------------------------------------------- usuários

pub async fn create_user(pool: &PgPool, id: Uuid, email: &str, password_hash: &str) -> Result<bool, sqlx::Error> {
    let r = sqlx::query("INSERT INTO users (id, email, password_hash) VALUES ($1, $2, $3) ON CONFLICT (email) DO NOTHING")
        .bind(id)
        .bind(email)
        .bind(password_hash)
        .execute(pool)
        .await?;
    Ok(r.rows_affected() == 1)
}

pub async fn find_user_by_email(pool: &PgPool, email: &str) -> Result<Option<UserRow>, sqlx::Error> {
    sqlx::query_as::<_, UserRow>("SELECT id, password_hash, is_active, totp_secret_enc FROM users WHERE email = $1")
    .bind(email)
    .fetch_optional(pool)
    .await
}

pub async fn find_email_by_id(pool: &PgPool, id: Uuid) -> Result<Option<(String, DateTime<Utc>)>, sqlx::Error> {
    sqlx::query_as::<_, (String, DateTime<Utc>)>("SELECT email, created_at FROM users WHERE id = $1 AND is_active")
        .bind(id)
        .fetch_optional(pool)
        .await
}

/// Situação de login de uma conta vista de uma origem.
#[derive(Debug, sqlx::FromRow)]
pub struct LoginGate {
    /// Esta origem errou demais: bloqueada (só ela).
    pub origin_locked: bool,
    /// A conta inteira está bloqueada (ataque distribuído)...
    pub account_locked: bool,
    /// ...mas esta origem já entrou com sucesso recentemente e é poupada.
    pub known_origin: bool,
}

impl LoginGate {
    pub fn blocked(&self) -> bool {
        self.origin_locked || (self.account_locked && !self.known_origin)
    }
}

pub async fn login_gate(pool: &PgPool, user_id: Uuid, origin: &str) -> Result<LoginGate, sqlx::Error> {
    sqlx::query_as::<_, LoginGate>(
        "SELECT
            COALESCE(a.locked_until > now(), false) AS origin_locked,
            COALESCE(u.locked_until > now(), false) AS account_locked,
            COALESCE(a.last_success_at > now() - make_interval(days => $3::int), false) AS known_origin
         FROM users u LEFT JOIN login_attempts a ON a.user_id = u.id AND a.origin = $2
         WHERE u.id = $1",
    )
    .bind(user_id)
    .bind(origin)
    .bind(KNOWN_ORIGIN_DAYS as i32)
    .fetch_one(pool)
    .await
}

/// Resultado de registrar uma falha.
#[derive(Debug, sqlx::FromRow)]
pub struct FailureOutcome {
    /// Esta falha fez a CONTA INTEIRA ser bloqueada agora (para alertar uma única vez).
    pub account_just_locked: bool,
}

/// Conta uma falha de login da origem e da conta. Bloqueio com espera exponencial
/// (60 s, 120 s, 240 s... até 1 h): a origem a partir de `MAX_FAILED_LOGINS`, a conta inteira a
/// partir de `ACCOUNT_MAX_FAILED_LOGINS`. Contadores sem falha há 24 h recomeçam do zero.
pub async fn register_login_failure(pool: &PgPool, user_id: Uuid, origin: &str) -> Result<FailureOutcome, sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO login_attempts AS a (user_id, origin, failed_attempts, last_failure_at)
         VALUES ($1, $2, 1, now())
         ON CONFLICT (user_id, origin) DO UPDATE SET
            failed_attempts = CASE WHEN a.last_failure_at < now() - interval '1 day' THEN 1 ELSE a.failed_attempts + 1 END,
            last_failure_at = now(),
            locked_until = CASE
                WHEN a.last_failure_at >= now() - interval '1 day' AND a.failed_attempts + 1 >= $3
                THEN now() + make_interval(secs => LEAST($4::float8 * power(2::float8, (a.failed_attempts + 1 - $3)::float8), 3600::float8))
                ELSE a.locked_until END",
    )
    .bind(user_id)
    .bind(origin)
    .bind(MAX_FAILED_LOGINS)
    .bind(LOCK_BASE_SECS)
    .execute(&mut *tx)
    .await?;
    let out = sqlx::query_as::<_, FailureOutcome>(
        "UPDATE users SET
            failed_attempts = CASE WHEN last_failed_at < now() - interval '1 day' THEN 1 ELSE failed_attempts + 1 END,
            locked_until = CASE
                WHEN COALESCE(last_failed_at >= now() - interval '1 day', true) AND failed_attempts + 1 >= $2
                THEN now() + make_interval(secs => LEAST($3::float8 * power(2::float8, (failed_attempts + 1 - $2)::float8), 3600::float8))
                ELSE locked_until END,
            last_failed_at = now()
         WHERE id = $1
         RETURNING (failed_attempts = $2) AS account_just_locked",
    )
    .bind(user_id)
    .bind(ACCOUNT_MAX_FAILED_LOGINS)
    .bind(LOCK_BASE_SECS)
    .fetch_one(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(out)
}

/// Login bem-sucedido: zera os contadores da origem e da conta e marca a origem como conhecida.
pub async fn register_login_success(pool: &PgPool, user_id: Uuid, origin: &str) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;
    sqlx::query(
        "INSERT INTO login_attempts (user_id, origin, last_success_at) VALUES ($1, $2, now())
         ON CONFLICT (user_id, origin) DO UPDATE SET failed_attempts = 0, locked_until = NULL, last_success_at = now()",
    )
    .bind(user_id)
    .bind(origin)
    .execute(&mut *tx)
    .await?;
    sqlx::query("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = $1 AND (failed_attempts <> 0 OR locked_until IS NOT NULL)")
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await
}

/// Remove todos os bloqueios do usuário (comando do operador).
pub async fn unlock_user(pool: &PgPool, user_id: Uuid) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = $1").bind(user_id).execute(pool).await?;
    sqlx::query("UPDATE login_attempts SET failed_attempts = 0, locked_until = NULL WHERE user_id = $1").bind(user_id).execute(pool).await?;
    Ok(())
}

pub async fn find_user_id_by_email(pool: &PgPool, email: &str) -> Result<Option<Uuid>, sqlx::Error> {
    sqlx::query_scalar("SELECT id FROM users WHERE email = $1").bind(email).fetch_optional(pool).await
}

// ---------------------------------------------------------------- MFA

pub async fn set_totp_secret(pool: &PgPool, user_id: Uuid, secret_enc: Option<&[u8]>) -> Result<bool, sqlx::Error> {
    let r = sqlx::query("UPDATE users SET totp_secret_enc = $2, totp_last_step = NULL WHERE id = $1")
        .bind(user_id)
        .bind(secret_enc)
        .execute(pool)
        .await?;
    Ok(r.rows_affected() == 1)
}

/// Registra o passo TOTP usado. `false` se ele (ou um posterior) já foi usado: código repetido.
pub async fn consume_totp_step(pool: &PgPool, user_id: Uuid, step: u64) -> Result<bool, sqlx::Error> {
    let r = sqlx::query("UPDATE users SET totp_last_step = $2 WHERE id = $1 AND (totp_last_step IS NULL OR totp_last_step < $2)")
        .bind(user_id)
        .bind(step as i64)
        .execute(pool)
        .await?;
    Ok(r.rows_affected() == 1)
}

// ---------------------------------------------------------------- revogação de access tokens

/// Estado que decide se um access token (JWT) ainda vale: usuário ativo e emitido depois do
/// último "derrubar sessões".
#[derive(Debug, sqlx::FromRow)]
pub struct TokenGate {
    pub is_active: bool,
    /// Segundos Unix (inteiros) do último corte, se houver.
    pub valid_after: Option<i64>,
}

pub async fn token_gate(pool: &PgPool, user_id: Uuid) -> Result<Option<TokenGate>, sqlx::Error> {
    sqlx::query_as::<_, TokenGate>(
        "SELECT is_active, floor(EXTRACT(EPOCH FROM tokens_valid_after))::int8 AS valid_after FROM users WHERE id = $1",
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await
}

/// Invalida na hora todos os access tokens já emitidos para o usuário.
pub async fn invalidate_access_tokens(pool: &PgPool, user_id: Uuid) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE users SET tokens_valid_after = now() WHERE id = $1").bind(user_id).execute(pool).await?;
    Ok(())
}

/// Derruba TODAS as sessões do usuário: access tokens e refresh tokens.
pub async fn revoke_all_sessions(pool: &PgPool, user_id: Uuid) -> Result<(), sqlx::Error> {
    invalidate_access_tokens(pool, user_id).await?;
    sqlx::query("UPDATE refresh_tokens SET revoked_at = now() WHERE user_id = $1 AND revoked_at IS NULL")
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}

// ---------------------------------------------------------------- clientes

pub async fn get_client(pool: &PgPool, client_id: &str) -> Result<Option<ClientRow>, sqlx::Error> {
    sqlx::query_as::<_, ClientRow>(
        "SELECT client_id, name, redirect_uris, allowed_scopes, is_active FROM oauth_clients WHERE client_id = $1",
    )
    .bind(client_id)
    .fetch_optional(pool)
    .await
}

pub async fn create_client(
    pool: &PgPool,
    client_id: &str,
    name: &str,
    redirect_uris: &[String],
    scopes: &[String],
) -> Result<bool, sqlx::Error> {
    let r = sqlx::query(
        "INSERT INTO oauth_clients (client_id, name, redirect_uris, allowed_scopes) VALUES ($1, $2, $3, $4) ON CONFLICT (client_id) DO NOTHING",
    )
    .bind(client_id)
    .bind(name)
    .bind(redirect_uris)
    .bind(scopes)
    .execute(pool)
    .await?;
    Ok(r.rows_affected() == 1)
}

/// Garante o cliente do Swagger UI (público, PKCE) apontando para a URL atual. Não reativa um
/// cliente que o operador tenha desativado.
pub async fn ensure_client(pool: &PgPool, client_id: &str, name: &str, redirect_uri: &str, scopes: &[String]) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO oauth_clients (client_id, name, redirect_uris, allowed_scopes) VALUES ($1, $2, ARRAY[$3], $4)
         ON CONFLICT (client_id) DO UPDATE SET redirect_uris = EXCLUDED.redirect_uris, allowed_scopes = EXCLUDED.allowed_scopes",
    )
    .bind(client_id)
    .bind(name)
    .bind(redirect_uri)
    .bind(scopes)
    .execute(pool)
    .await?;
    Ok(())
}

// ---------------------------------------------------------- authorization code

#[allow(clippy::too_many_arguments)] // espelha as colunas da tabela
pub async fn insert_auth_code(
    pool: &PgPool,
    code_hash: &[u8],
    client_id: &str,
    user_id: Uuid,
    redirect_uri: &str,
    scope: &str,
    code_challenge: &str,
    ttl_secs: f64,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO auth_codes (code_hash, client_id, user_id, redirect_uri, scope, code_challenge, expires_at)
         VALUES ($1, $2, $3, $4, $5, $6, now() + make_interval(secs => $7::float8))",
    )
    .bind(code_hash)
    .bind(client_id)
    .bind(user_id)
    .bind(redirect_uri)
    .bind(scope)
    .bind(code_challenge)
    .bind(ttl_secs)
    .execute(pool)
    .await?;
    Ok(())
}

/// Consome o código de forma atômica: só o primeiro `UPDATE` vence.
pub async fn consume_auth_code(pool: &PgPool, code_hash: &[u8]) -> Result<CodeOutcome, sqlx::Error> {
    let row = sqlx::query_as::<_, AuthCodeRow>(
        "UPDATE auth_codes SET used_at = now()
         WHERE code_hash = $1 AND used_at IS NULL
         RETURNING client_id, user_id, redirect_uri, scope, code_challenge, (expires_at <= now()) AS expired",
    )
    .bind(code_hash)
    .fetch_optional(pool)
    .await?;
    if let Some(r) = row {
        return Ok(CodeOutcome::Valid(r));
    }
    let prior = sqlx::query_as::<_, (Uuid, String)>("SELECT user_id, client_id FROM auth_codes WHERE code_hash = $1")
        .bind(code_hash)
        .fetch_optional(pool)
        .await?;
    Ok(match prior {
        Some((user_id, client_id)) => CodeOutcome::Reused { user_id, client_id },
        None => CodeOutcome::Unknown,
    })
}

// ---------------------------------------------------------------- refresh tokens

pub async fn insert_refresh_new_family(
    pool: &PgPool,
    token_hash: &[u8],
    user_id: Uuid,
    client_id: &str,
    scope: &str,
    ttl_secs: f64,
    family_max_secs: f64,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO refresh_tokens (id, token_hash, family_id, family_started_at, user_id, client_id, scope, expires_at)
         VALUES ($1, $2, $3, now(), $4, $5, $6, LEAST(now() + make_interval(secs => $7::float8), now() + make_interval(secs => $8::float8)))",
    )
    .bind(Uuid::new_v4())
    .bind(token_hash)
    .bind(Uuid::new_v4())
    .bind(user_id)
    .bind(client_id)
    .bind(scope)
    .bind(ttl_secs)
    .bind(family_max_secs)
    .execute(pool)
    .await?;
    Ok(())
}

#[derive(sqlx::FromRow)]
struct RefreshLookup {
    id: Uuid,
    family_id: Uuid,
    user_id: Uuid,
    client_id: String,
    scope: String,
    expired: bool,
    used: bool,
    revoked: bool,
    family_expired: bool,
    user_active: bool,
    family_remaining_secs: f64,
}

fn scope_is_subset(requested: &str, granted: &str) -> bool {
    requested
        .split_ascii_whitespace()
        .all(|r| granted.split_ascii_whitespace().any(|g| g == r))
}

pub async fn rotate_refresh(
    pool: &PgPool,
    token_hash: &[u8],
    client_id: &str,
    new_hash: &[u8],
    requested_scope: Option<&str>,
    ttl_secs: f64,
    family_max_secs: f64,
) -> Result<RotateOutcome, sqlx::Error> {
    let mut tx = pool.begin().await?;

    let Some(t) = sqlx::query_as::<_, RefreshLookup>(
        "SELECT r.id, r.family_id, r.user_id, r.client_id, r.scope,
                (r.expires_at <= now()) AS expired,
                (r.used_at IS NOT NULL) AS used,
                (r.revoked_at IS NOT NULL) AS revoked,
                (r.family_started_at + make_interval(secs => $2::float8) <= now()) AS family_expired,
                u.is_active AS user_active,
                EXTRACT(EPOCH FROM (r.family_started_at + make_interval(secs => $2::float8) - now()))::float8 AS family_remaining_secs
         FROM refresh_tokens r JOIN users u ON u.id = r.user_id
         WHERE r.token_hash = $1
         FOR UPDATE OF r",
    )
    .bind(token_hash)
    .bind(family_max_secs)
    .fetch_optional(&mut *tx)
    .await?
    else {
        return Ok(RotateOutcome::Invalid);
    };

    // Token apresentado por outro cliente, ou usuário desativado: revoga a família.
    if t.client_id != client_id || !t.user_active {
        revoke_family_tx(&mut tx, t.family_id).await?;
        tx.commit().await?;
        return Ok(RotateOutcome::Invalid);
    }
    // Reutilização de token já rotacionado => assume roubo, revoga tudo.
    if t.used {
        revoke_family_tx(&mut tx, t.family_id).await?;
        tx.commit().await?;
        return Ok(RotateOutcome::Reuse { user_id: t.user_id });
    }
    if t.revoked || t.expired || t.family_expired {
        tx.rollback().await?;
        return Ok(RotateOutcome::Invalid);
    }

    let scope = match requested_scope {
        None => t.scope.clone(),
        Some(req) if scope_is_subset(req, &t.scope) => req.to_string(),
        Some(_) => {
            tx.rollback().await?;
            return Ok(RotateOutcome::InvalidScope);
        }
    };

    sqlx::query("UPDATE refresh_tokens SET used_at = now() WHERE id = $1")
        .bind(t.id)
        .execute(&mut *tx)
        .await?;

    let new_ttl = ttl_secs.min(t.family_remaining_secs.max(1.0));
    sqlx::query(
        "INSERT INTO refresh_tokens (id, token_hash, family_id, family_started_at, user_id, client_id, scope, expires_at)
         SELECT $1, $2, family_id, family_started_at, user_id, client_id, $3, now() + make_interval(secs => $4::float8)
         FROM refresh_tokens WHERE id = $5",
    )
    .bind(Uuid::new_v4())
    .bind(new_hash)
    .bind(&scope)
    .bind(new_ttl)
    .bind(t.id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(RotateOutcome::Rotated { user_id: t.user_id, scope })
}

async fn revoke_family_tx(tx: &mut sqlx::PgConnection, family_id: Uuid) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE refresh_tokens SET revoked_at = now() WHERE family_id = $1 AND revoked_at IS NULL")
        .bind(family_id)
        .execute(tx)
        .await?;
    Ok(())
}

/// RFC 7009: revoga a família do refresh token, se existir e for do cliente.
pub async fn revoke_by_token(pool: &PgPool, token_hash: &[u8], client_id: &str) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE refresh_tokens SET revoked_at = now()
         WHERE revoked_at IS NULL AND family_id IN
               (SELECT family_id FROM refresh_tokens WHERE token_hash = $1 AND client_id = $2)",
    )
    .bind(token_hash)
    .bind(client_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn revoke_user_client(pool: &PgPool, user_id: Uuid, client_id: &str) -> Result<(), sqlx::Error> {
    sqlx::query("UPDATE refresh_tokens SET revoked_at = now() WHERE user_id = $1 AND client_id = $2 AND revoked_at IS NULL")
        .bind(user_id)
        .bind(client_id)
        .execute(pool)
        .await?;
    Ok(())
}

// ---------------------------------------------------------------- auditoria / limpeza

pub async fn audit(pool: &PgPool, kind: &str, user_id: Option<Uuid>, client_id: Option<&str>, ip: Option<&str>, detail: Option<&str>) {
    let r = sqlx::query("INSERT INTO audit_log (kind, user_id, client_id, ip, detail) VALUES ($1, $2, $3, $4, $5)")
        .bind(kind)
        .bind(user_id)
        .bind(client_id)
        .bind(ip)
        .bind(detail)
        .execute(pool)
        .await;
    if let Err(e) = r {
        tracing::warn!(error = %e, "falha ao gravar auditoria");
    }
}

pub async fn cleanup(pool: &PgPool) -> Result<u64, sqlx::Error> {
    let mut n = 0;
    n += sqlx::query("DELETE FROM auth_codes WHERE expires_at < now() - interval '1 day'")
        .execute(pool)
        .await?
        .rows_affected();
    n += sqlx::query("DELETE FROM refresh_tokens WHERE expires_at < now() - interval '7 days'")
        .execute(pool)
        .await?
        .rows_affected();
    n += sqlx::query(
        "DELETE FROM login_attempts WHERE COALESCE(last_failure_at, 'epoch') < now() - interval '30 days'
                                     AND COALESCE(last_success_at, 'epoch') < now() - interval '90 days'",
    )
    .execute(pool)
    .await?
    .rows_affected();
    n += sqlx::query("DELETE FROM audit_log WHERE at < now() - interval '180 days'")
        .execute(pool)
        .await?
        .rows_affected();
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::scope_is_subset;

    #[test]
    fn scope_subset_rules() {
        assert!(scope_is_subset("ocr", "ocr profile"));
        assert!(scope_is_subset("", "ocr"));
        assert!(!scope_is_subset("admin", "ocr profile"));
        assert!(!scope_is_subset("ocr admin", "ocr"));
        // não confunde prefixo com escopo
        assert!(!scope_is_subset("oc", "ocr"));
    }
}
