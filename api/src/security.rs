//! Eventos de segurança: trilha no banco (`audit_log`) + log estruturado + alertas.
//!
//! Todo evento vai para o log com `target = "security"` (campos `kind`, `user_id`, `client_id`,
//! `ip`, `detail`), em JSON com `LOG_FORMAT=json`. Situações que pedem gente olhando geram, além
//! disso, um evento `target = "security_alert"` no nível ERROR — é nele que o coletor de logs
//! (Loki, Datadog, CloudWatch...) deve pendurar o alerta:
//!
//! * reutilização de refresh token ou de authorization code (indício de roubo de token);
//! * bloqueio de conta inteira (ataque distribuído contra um usuário);
//! * worker de OCR morto por sinal (indício de arquivo tentando explorar o parser);
//! * muitas falhas de login vindas de um mesmo IP numa janela de 5 minutos.
//!
//! A contagem por IP é por réplica (memória local); o `audit_log` é a visão consolidada.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use uuid::Uuid;

use crate::state::AppState;
use crate::store;

const WINDOW: Duration = Duration::from_secs(300);
const MAX_TRACKED_IPS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warn,
    Alert,
}

pub fn severity(kind: &str) -> Severity {
    match kind {
        "refresh_reuse_detected" | "auth_code_reuse" | "account_locked" | "ocr_worker_crashed" | "chunk_worker_crashed" | "embedding_worker_crashed" => {
            Severity::Alert
        }
        "login_failed" | "login_failed_unknown" | "login_blocked" | "login_ip_locked" | "mfa_failed"
        | "token_exchange_failed" | "ocr_worker_failed" | "ocr_worker_bad_reply" | "ocr_worker_memory_killed"
        | "chunk_worker_failed" | "chunk_worker_bad_reply" | "chunk_worker_memory_killed" | "embedding_worker_failed"
        | "embedding_worker_bad_reply" | "embedding_worker_memory_killed" | "sessions_revoked" => Severity::Warn,
        _ => Severity::Info,
    }
}

fn counts_as_login_failure(kind: &str) -> bool {
    matches!(kind, "login_failed" | "login_failed_unknown" | "login_blocked" | "login_ip_locked" | "mfa_failed")
}

struct Window {
    started: Instant,
    count: u32,
    alerted: bool,
}

/// Conta falhas de login por IP numa janela fixa e avisa UMA vez por janela ao passar do limiar.
pub struct Monitor {
    threshold: u32,
    per_ip: Mutex<HashMap<String, Window>>,
}

impl Monitor {
    pub fn new(threshold: u32) -> Self {
        Self { threshold: threshold.max(1), per_ip: Mutex::new(HashMap::new()) }
    }

    /// `Some(n)` quando este evento fez o IP cruzar o limiar (n = falhas na janela).
    pub fn observe_login_failure(&self, ip: &str, now: Instant) -> Option<u32> {
        let mut map = self.per_ip.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.len() >= MAX_TRACKED_IPS && !map.contains_key(ip) {
            map.retain(|_, w| now.duration_since(w.started) < WINDOW);
            if map.len() >= MAX_TRACKED_IPS {
                return None; // sob inundação de IPs: o limite de taxa já está agindo
            }
        }
        let w = map.entry(ip.to_string()).or_insert(Window { started: now, count: 0, alerted: false });
        if now.duration_since(w.started) >= WINDOW {
            *w = Window { started: now, count: 0, alerted: false };
        }
        w.count += 1;
        if w.count >= self.threshold && !w.alerted {
            w.alerted = true;
            return Some(w.count);
        }
        None
    }

    pub fn prune(&self) {
        let now = Instant::now();
        let mut map = self.per_ip.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        map.retain(|_, w| now.duration_since(w.started) < WINDOW);
        map.shrink_to_fit();
    }
}

fn log(kind: &str, user: Option<Uuid>, client: Option<&str>, ip: Option<&str>, detail: Option<&str>) {
    let user = user.map(|u| u.to_string());
    let (user, client, ip, detail) = (user.as_deref().unwrap_or("-"), client.unwrap_or("-"), ip.unwrap_or("-"), detail.unwrap_or(""));
    match severity(kind) {
        Severity::Info => tracing::info!(target: "security", kind, user_id = user, client_id = client, ip, detail, "evento de segurança"),
        Severity::Warn | Severity::Alert => {
            tracing::warn!(target: "security", kind, user_id = user, client_id = client, ip, detail, "evento de segurança")
        }
    }
}

fn alert(kind: &str, user: Option<Uuid>, ip: Option<&str>, reason: &str) {
    let user = user.map(|u| u.to_string());
    tracing::error!(
        target: "security_alert",
        kind,
        user_id = user.as_deref().unwrap_or("-"),
        ip = ip.unwrap_or("-"),
        reason,
        "ALERTA DE SEGURANÇA"
    );
}

/// Registra um evento: log estruturado, trilha no banco e, se for o caso, alerta.
pub async fn event(st: &AppState, kind: &str, user: Option<Uuid>, client: Option<&str>, ip: Option<&str>, detail: Option<&str>) {
    log(kind, user, client, ip, detail);
    store::audit(&st.db, kind, user, client, ip, detail).await;

    let mut reason: Option<String> = None;
    if severity(kind) == Severity::Alert {
        reason = Some(detail.unwrap_or(kind).to_string());
    } else if counts_as_login_failure(kind)
        && let Some(ip) = ip
        && let Some(n) = st.security.observe_login_failure(ip, Instant::now())
    {
        reason = Some(format!("{n} falhas de login do mesmo IP em 5 minutos"));
    }
    if let Some(r) = reason {
        alert(kind, user, ip, &r);
        store::audit(&st.db, "security_alert", user, client, ip, Some(&format!("{kind}: {r}"))).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alerts_once_per_window_per_ip() {
        let m = Monitor::new(3);
        let t0 = Instant::now();
        assert_eq!(m.observe_login_failure("1.1.1.1", t0), None);
        assert_eq!(m.observe_login_failure("1.1.1.1", t0), None);
        assert_eq!(m.observe_login_failure("1.1.1.1", t0), Some(3));
        assert_eq!(m.observe_login_failure("1.1.1.1", t0), None, "só um alerta por janela");
        assert_eq!(m.observe_login_failure("2.2.2.2", t0), None, "outro IP tem contagem própria");
        // Nova janela: conta de novo.
        let t1 = t0 + WINDOW + Duration::from_secs(1);
        assert_eq!(m.observe_login_failure("1.1.1.1", t1), None);
        assert_eq!(m.observe_login_failure("1.1.1.1", t1), None);
        assert_eq!(m.observe_login_failure("1.1.1.1", t1), Some(3));
    }

    #[test]
    fn theft_indicators_are_alerts() {
        for k in ["refresh_reuse_detected", "auth_code_reuse", "account_locked", "ocr_worker_crashed", "chunk_worker_crashed", "embedding_worker_crashed"] {
            assert_eq!(severity(k), Severity::Alert, "{k}");
        }
        assert_eq!(severity("login_ok"), Severity::Info);
        assert_eq!(severity("login_failed"), Severity::Warn);
    }
}
