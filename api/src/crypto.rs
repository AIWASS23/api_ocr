use argon2::password_hash::phc::PasswordHash;
use argon2::password_hash::{PasswordHasher, PasswordVerifier};
use argon2::{Algorithm, Argon2, Params, Version};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::Semaphore;

/// `n` bytes aleatórios de um CSPRNG, em base64url (sem padding).
pub fn random_token(n: usize) -> String {
    let mut buf = vec![0u8; n];
    rand::fill(&mut buf[..]);
    URL_SAFE_NO_PAD.encode(buf)
}

/// Token opaco de 256 bits.
pub fn new_opaque_token() -> String {
    random_token(32)
}

pub fn sha256(data: &[u8]) -> Vec<u8> {
    Sha256::digest(data).to_vec()
}

/// Hash de um token para guardar no banco (tokens têm 256 bits de entropia,
/// então SHA-256 sem sal é suficiente e permite busca indexada).
pub fn hash_token(token: &str) -> Vec<u8> {
    sha256(token.as_bytes())
}

/// Comparação em tempo constante.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}

/// PKCE S256: base64url(sha256(verifier)) == challenge.
pub fn pkce_matches(verifier: &str, challenge: &str) -> bool {
    let computed = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    ct_eq(computed.as_bytes(), challenge.as_bytes())
}

#[derive(Debug, thiserror::Error)]
pub enum HashError {
    #[error("servidor ocupado")]
    Busy,
    #[error("falha ao calcular hash")]
    Internal,
}

/// Argon2id com pepper. O hash roda em thread de bloqueio e sob um semáforo,
/// porque é caro de propósito e seria um vetor de DoS se ilimitado.
pub struct PasswordService {
    pepper: Vec<u8>,
    gate: Semaphore,
    dummy_hash: String,
}

// OWASP: Argon2id com m=19 MiB, t=2, p=1 é o mínimo; usamos 64 MiB.
const M_COST_KIB: u32 = 64 * 1024;
const T_COST: u32 = 3;
const P_COST: u32 = 1;

fn argon(pepper: &[u8]) -> Result<Argon2<'_>, HashError> {
    let params = Params::new(M_COST_KIB, T_COST, P_COST, None).map_err(|_| HashError::Internal)?;
    Argon2::new_with_secret(pepper, Algorithm::Argon2id, Version::V0x13, params)
        .map_err(|_| HashError::Internal)
}

fn hash_blocking(pepper: &[u8], password: &[u8]) -> Result<String, HashError> {
    let h = argon(pepper)?
        .hash_password(password)
        .map_err(|_| HashError::Internal)?;
    Ok(h.to_string())
}

fn verify_blocking(pepper: &[u8], password: &[u8], phc: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(phc) else {
        return false;
    };
    let Ok(a) = argon(pepper) else { return false };
    a.verify_password(password, &parsed).is_ok()
}

impl PasswordService {
    pub fn new(pepper: Vec<u8>, max_concurrent: usize) -> Result<Self, HashError> {
        // Hash falso para igualar o tempo de resposta quando o usuário não existe.
        let dummy_hash = hash_blocking(&pepper, b"dummy-password-for-timing")?;
        Ok(Self {
            pepper,
            gate: Semaphore::new(max_concurrent.max(1)),
            dummy_hash,
        })
    }

    async fn permit(&self) -> Result<tokio::sync::SemaphorePermit<'_>, HashError> {
        match tokio::time::timeout(std::time::Duration::from_secs(3), self.gate.acquire()).await {
            Ok(Ok(p)) => Ok(p),
            _ => Err(HashError::Busy),
        }
    }

    pub async fn hash(&self, password: String) -> Result<String, HashError> {
        let _permit = self.permit().await?;
        let pepper = self.pepper.clone();
        tokio::task::spawn_blocking(move || hash_blocking(&pepper, password.as_bytes()))
            .await
            .map_err(|_| HashError::Internal)?
    }

    /// `phc = None` verifica contra um hash falso (tempo constante) e devolve `false`.
    pub async fn verify(&self, password: String, phc: Option<String>) -> Result<bool, HashError> {
        let _permit = self.permit().await?;
        let pepper = self.pepper.clone();
        let known = phc.is_some();
        let phc = phc.unwrap_or_else(|| self.dummy_hash.clone());
        let ok = tokio::task::spawn_blocking(move || verify_blocking(&pepper, password.as_bytes(), &phc))
            .await
            .map_err(|_| HashError::Internal)?;
        Ok(ok && known)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_rfc7636_vector() {
        // Exemplo do Apêndice B da RFC 7636.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";
        assert!(pkce_matches(verifier, challenge));
        assert!(!pkce_matches("outro-verifier", challenge));
    }

    #[test]
    fn tokens_are_unique_and_long() {
        let a = new_opaque_token();
        let b = new_opaque_token();
        assert_ne!(a, b);
        assert_eq!(a.len(), 43);
    }

    #[tokio::test]
    async fn password_roundtrip_and_unknown_user() {
        let svc = PasswordService::new(vec![7u8; 32], 2).unwrap();
        let h = svc.hash("senha-bem-longa-123".into()).await.unwrap();
        assert!(h.starts_with("$argon2id$"));
        assert!(svc.verify("senha-bem-longa-123".into(), Some(h.clone())).await.unwrap());
        assert!(!svc.verify("errada".into(), Some(h)).await.unwrap());
        // usuário inexistente: nunca autentica, mesmo com "senha correta"
        assert!(!svc.verify("senha-bem-longa-123".into(), None).await.unwrap());
    }
}
