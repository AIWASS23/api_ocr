use std::collections::HashSet;
use std::ops::Deref;
use std::sync::{Arc, Mutex, PoisonError};

use sqlx::PgPool;
use tokio::sync::Semaphore;
use uuid::Uuid;

use crate::config::Config;
use crate::crypto::PasswordService;
use crate::limits::Limiters;
use crate::memory::MemoryGuard;
use crate::mfa::MfaCipher;
use crate::security::Monitor;
use crate::tokens::TokenService;

pub struct OcrRuntime {
    /// Quantos documentos são processados ao mesmo tempo em todo o servidor.
    pub slots: Arc<Semaphore>,
    in_flight: Mutex<HashSet<Uuid>>,
}

/// Garante no máximo 1 OCR simultâneo por usuário.
pub struct UserGuard {
    runtime: Arc<AppInner>,
    user: Uuid,
}

impl Drop for UserGuard {
    fn drop(&mut self) {
        self.runtime
            .ocr
            .in_flight
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.user);
    }
}

impl OcrRuntime {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(max_concurrent)),
            in_flight: Mutex::new(HashSet::new()),
        }
    }
}

pub struct AppInner {
    pub cfg: Config,
    pub db: PgPool,
    pub passwords: PasswordService,
    pub tokens: TokenService,
    pub limiters: Limiters,
    pub ocr: OcrRuntime,
    pub memory: MemoryGuard,
    pub security: Monitor,
    pub mfa: MfaCipher,
}

#[derive(Clone)]
pub struct AppState(pub Arc<AppInner>);

impl Deref for AppState {
    type Target = AppInner;
    fn deref(&self) -> &AppInner {
        &self.0
    }
}

impl AppState {
    pub fn new(inner: AppInner) -> Self {
        Self(Arc::new(inner))
    }

    /// `None` se o usuário já tem um OCR em andamento.
    pub fn try_user_guard(&self, user: Uuid) -> Option<UserGuard> {
        let mut set = self.0.ocr.in_flight.lock().unwrap_or_else(PoisonError::into_inner);
        if !set.insert(user) {
            return None;
        }
        Some(UserGuard { runtime: self.0.clone(), user })
    }
}
