//! Limitador de taxa compartilhado entre réplicas (Redis).
//!
//! Token bucket atômico em Lua: o tempo vem do próprio Redis (`TIME`), então as réplicas
//! não precisam de relógios sincronizados. É uma segunda camada, depois do limitador local:
//! se o Redis falhar, o tráfego continua protegido pelo limite local de cada réplica
//! (falha aberta, com disjuntor para não somar latência a cada requisição).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use redis::Script;

const LUA: &str = r#"
local rate = tonumber(ARGV[1])   -- tokens por ms
local burst = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local d = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(d[1])
local ts = tonumber(d[2])
if tokens == nil then tokens = burst; ts = now end
tokens = math.min(burst, tokens + math.max(0, now - ts) * rate)
local allowed = 0
local retry = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry = math.ceil((1 - tokens) / rate)
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(burst / rate) + 1000)
return {allowed, retry}
"#;

const CALL_TIMEOUT: Duration = Duration::from_millis(100);
const BREAKER_OPEN: Duration = Duration::from_secs(5);

pub struct SharedLimiter {
    conn: ConnectionManager,
    script: Script,
    prefix: String,
    /// Instante (ms desde a época) até o qual o Redis é ignorado após uma falha.
    down_until_ms: AtomicU64,
    warned: AtomicBool,
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

impl SharedLimiter {
    pub async fn connect(url: &str, prefix: &str) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(url)?;
        let cfg = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(Duration::from_secs(2)))
            .set_response_timeout(Some(CALL_TIMEOUT))
            .set_number_of_retries(1);
        let conn = ConnectionManager::new_with_config(client, cfg).await?;
        Ok(Self {
            conn,
            script: Script::new(LUA),
            prefix: prefix.to_string(),
            down_until_ms: AtomicU64::new(0),
            warned: AtomicBool::new(false),
        })
    }

    /// `Err(espera)` quando o limite global foi excedido. Falhas do Redis => `Ok` (falha aberta).
    pub async fn check(&self, scope: &str, key: &str, per_minute: u32, burst: u32) -> Result<(), Duration> {
        if now_ms() < self.down_until_ms.load(Ordering::Relaxed) {
            return Ok(());
        }
        let rate_per_ms = f64::from(per_minute.max(1)) / 60_000.0;
        let redis_key = format!("{}rl:{scope}:{key}", self.prefix);
        let mut conn = self.conn.clone();
        let mut invocation = self.script.key(redis_key);
        invocation.arg(rate_per_ms).arg(burst.max(1));
        let call = invocation.invoke_async::<(i64, i64)>(&mut conn);

        match tokio::time::timeout(CALL_TIMEOUT * 2, call).await {
            Ok(Ok((1, _))) => {
                self.warned.store(false, Ordering::Relaxed);
                Ok(())
            }
            Ok(Ok((_, retry_ms))) => Err(Duration::from_millis(retry_ms.max(1) as u64)),
            Ok(Err(e)) => {
                self.trip(&e.to_string());
                Ok(())
            }
            Err(_) => {
                self.trip("timeout");
                Ok(())
            }
        }
    }

    fn trip(&self, why: &str) {
        self.down_until_ms.store(now_ms() + BREAKER_OPEN.as_millis() as u64, Ordering::Relaxed);
        if !self.warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(motivo = why, "Redis do limitador indisponível: usando só o limite local por réplica");
        }
    }
}
