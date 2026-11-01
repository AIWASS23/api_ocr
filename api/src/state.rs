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

/// Vagas de um tipo de job que roda em processo worker (OCR, chunking).
pub struct JobRuntime {
    /// Quantos jobs deste tipo rodam ao mesmo tempo em todo o servidor.
    pub slots: Arc<Semaphore>,
    in_flight: Arc<Mutex<HashSet<Uuid>>>,
}

/// Garante no máximo 1 job simultâneo (do mesmo tipo) por usuário.
pub struct UserGuard {
    in_flight: Arc<Mutex<HashSet<Uuid>>>,
    user: Uuid,
}

impl Drop for UserGuard {
    fn drop(&mut self) {
        self.in_flight.lock().unwrap_or_else(PoisonError::into_inner).remove(&self.user);
    }
}

impl JobRuntime {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(max_concurrent)),
            in_flight: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// `None` se o usuário já tem um job deste tipo em andamento.
    pub fn try_user_guard(&self, user: Uuid) -> Option<UserGuard> {
        if !self.in_flight.lock().unwrap_or_else(PoisonError::into_inner).insert(user) {
            return None;
        }
        Some(UserGuard { in_flight: self.in_flight.clone(), user })
    }
}

pub struct AppInner {
    pub cfg: Config,
    pub db: PgPool,
    pub passwords: PasswordService,
    pub tokens: TokenService,
    pub limiters: Limiters,
    pub ocr: JobRuntime,
    pub chunk: JobRuntime,
    pub embedding: JobRuntime,
    /// Catálogo de estratégias de embedding, com a disponibilidade deste servidor.
    pub embedding_catalog: crate::embed::Catalog,
    /// Catálogo de estratégias de chunking, com a disponibilidade deste servidor.
    pub chunk_catalog: crate::chunk::Catalog,
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
        self.0.ocr.try_user_guard(user)
    }
}
