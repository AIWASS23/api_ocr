//! Limitação de taxa por chave (IP ou usuário) e extração do IP do cliente.

use std::net::{IpAddr, SocketAddr};
use std::num::NonZeroU32;
use std::time::Duration;

use axum::extract::ConnectInfo;
use axum::http::Request;
use governor::clock::{Clock, DefaultClock};
use governor::{DefaultKeyedRateLimiter, Quota, RateLimiter};

pub struct KeyedLimiter<K: std::hash::Hash + Eq + Clone> {
    inner: DefaultKeyedRateLimiter<K>,
    clock: DefaultClock,
}

impl<K: std::hash::Hash + Eq + Clone> KeyedLimiter<K> {
    /// `per_minute` requisições por minuto, com rajada de até `burst`.
    pub fn new(per_minute: u32, burst: u32) -> Self {
        let per_minute = NonZeroU32::new(per_minute.max(1)).expect("maior que zero");
        let burst = NonZeroU32::new(burst.max(1)).expect("maior que zero");
        Self {
            inner: RateLimiter::keyed(Quota::per_minute(per_minute).allow_burst(burst)),
            clock: DefaultClock::default(),
        }
    }

    /// `Err(espera)` quando o limite foi excedido.
    pub fn check(&self, key: &K) -> Result<(), Duration> {
        self.inner
            .check_key(key)
            .map_err(|not_until| not_until.wait_time_from(self.clock.now()))
    }

    /// Descarta chaves já "esquecidas" para a memória não crescer sem limite.
    pub fn prune(&self) {
        self.inner.retain_recent();
        self.inner.shrink_to_fit();
    }
}

pub struct Limiters {
    /// Todas as rotas, por IP.
    general: KeyedLimiter<IpAddr>,
    /// Rotas de autenticação (login, token, registro), por IP.
    auth: KeyedLimiter<IpAddr>,
    /// OCR, por usuário.
    ocr: KeyedLimiter<uuid::Uuid>,
    /// Chunking, por usuário.
    chunk: KeyedLimiter<uuid::Uuid>,
    quotas: [(u32, u32); 4],
    shared: Option<crate::shared_limit::SharedLimiter>,
}

fn burst_of(kind: usize, per_min: u32) -> u32 {
    match kind {
        0 => (per_min / 5).max(10),
        1 => (per_min / 2).max(5),
        2 => (per_min / 2).max(2),
        _ => (per_min / 4).max(5),
    }
}

impl Limiters {
    pub fn new(general_per_min: u32, auth_per_min: u32, ocr_per_min: u32, chunk_per_min: u32) -> Self {
        let per = [general_per_min, auth_per_min, ocr_per_min, chunk_per_min];
        let quotas: [(u32, u32); 4] = std::array::from_fn(|k| (per[k], burst_of(k, per[k])));
        Self {
            general: KeyedLimiter::new(quotas[0].0, quotas[0].1),
            auth: KeyedLimiter::new(quotas[1].0, quotas[1].1),
            ocr: KeyedLimiter::new(quotas[2].0, quotas[2].1),
            chunk: KeyedLimiter::new(quotas[3].0, quotas[3].1),
            quotas,
            shared: None,
        }
    }

    /// Liga o limite global compartilhado entre réplicas (Redis).
    pub fn with_shared(mut self, shared: crate::shared_limit::SharedLimiter) -> Self {
        self.shared = Some(shared);
        self
    }

    async fn global(&self, kind: usize, scope: &str, key: &str) -> Result<(), Duration> {
        match &self.shared {
            Some(s) => s.check(scope, key, self.quotas[kind].0, self.quotas[kind].1).await,
            None => Ok(()),
        }
    }

    /// Primeiro o limite local (barato, sem rede); depois o global, se configurado.
    pub async fn check_general(&self, ip: IpAddr) -> Result<(), Duration> {
        self.general.check(&ip)?;
        self.global(0, "general", &ip.to_string()).await
    }

    pub async fn check_auth(&self, ip: IpAddr) -> Result<(), Duration> {
        self.auth.check(&ip)?;
        self.global(1, "auth", &ip.to_string()).await
    }

    pub async fn check_ocr(&self, user: uuid::Uuid) -> Result<(), Duration> {
        self.ocr.check(&user)?;
        self.global(2, "ocr", &user.to_string()).await
    }

    pub async fn check_chunk(&self, user: uuid::Uuid) -> Result<(), Duration> {
        self.chunk.check(&user)?;
        self.global(3, "chunk", &user.to_string()).await
    }

    pub fn prune(&self) {
        self.general.prune();
        self.auth.prune();
        self.ocr.prune();
        self.chunk.prune();
    }
}

/// IP do cliente. Sem proxy confiável, é o par TCP. Com `trust_proxy`, usa o
/// ÚLTIMO valor de `X-Forwarded-For` (o que o nosso proxy acrescentou — os
/// anteriores podem ser forjados pelo cliente).
pub fn client_ip<B>(req: &Request<B>, trust_proxy: bool) -> IpAddr {
    client_ip_from(req.extensions(), req.headers(), trust_proxy)
}

pub fn client_ip_from(ext: &axum::http::Extensions, headers: &axum::http::HeaderMap, trust_proxy: bool) -> IpAddr {
    let peer = ext
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip())
        .unwrap_or(IpAddr::from([0, 0, 0, 0]));
    if !trust_proxy {
        return peer;
    }
    headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.rsplit(',').next())
        .and_then(|s| s.trim().parse::<IpAddr>().ok())
        .unwrap_or(peer)
}

/// Chave de "origem" para o bloqueio de login: o IPv4 inteiro, ou o prefixo /64 do IPv6 (um
/// único cliente costuma ter um /64 inteiro; contar por endereço deixaria trocar de IP à vontade).
pub fn origin_key(ip: IpAddr) -> String {
    match ip {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => v4.to_string(),
            None => {
                let s = v6.segments();
                format!("{:x}:{:x}:{:x}:{:x}::/64", s[0], s[1], s[2], s[3])
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_groups_ipv6_by_64_and_unmaps_ipv4() {
        assert_eq!(origin_key("203.0.113.9".parse().unwrap()), "203.0.113.9");
        assert_eq!(origin_key("::ffff:203.0.113.9".parse().unwrap()), "203.0.113.9");
        let a = origin_key("2001:db8:1:2::1".parse().unwrap());
        let b = origin_key("2001:db8:1:2:ffff:ffff:ffff:ffff".parse().unwrap());
        assert_eq!(a, b);
        assert_eq!(a, "2001:db8:1:2::/64");
        assert_ne!(a, origin_key("2001:db8:1:3::1".parse().unwrap()));
    }

    #[test]
    fn blocks_after_burst_and_isolates_keys() {
        let l: KeyedLimiter<u8> = KeyedLimiter::new(60, 3);
        for _ in 0..3 {
            assert!(l.check(&1).is_ok());
        }
        let wait = l.check(&1).expect_err("4ª deve bloquear");
        assert!(wait <= Duration::from_secs(2));
        assert!(l.check(&2).is_ok(), "outra chave não é afetada");
    }

    #[test]
    fn xff_uses_last_hop_only_when_trusted() {
        let mut req = Request::builder()
            .header("x-forwarded-for", "6.6.6.6, 203.0.113.9")
            .body(())
            .unwrap();
        req.extensions_mut().insert(ConnectInfo(SocketAddr::from(([10, 0, 0, 1], 5555))));
        assert_eq!(client_ip(&req, true), IpAddr::from([203, 0, 113, 9]));
        assert_eq!(client_ip(&req, false), IpAddr::from([10, 0, 0, 1]));
    }
}
