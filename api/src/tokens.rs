use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation, decode, decode_header, encode};
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const AUDIENCE: &str = "api-ocr";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub iss: String,
    pub aud: String,
    pub sub: String,
    pub scope: String,
    pub client_id: String,
    pub exp: u64,
    pub iat: u64,
    pub nbf: u64,
    pub jti: String,
}

impl Claims {
    pub fn has_scope(&self, wanted: &str) -> bool {
        self.scope.split_ascii_whitespace().any(|s| s == wanted)
    }
}

/// Identificador público de uma chave: 16 hex de SHA-256 com separação de domínio.
/// Não revela nada útil do segredo (que tem >= 256 bits de entropia).
pub fn key_id(secret: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b"api_ocr jwt kid v1\0");
    h.update(secret);
    hex::encode(&h.finalize()[..8])
}

/// Emite e valida access tokens JWT (HS256, algoritmo fixo: um token com
/// `alg` diferente — inclusive `none` — é rejeitado antes de qualquer verificação).
///
/// Rotação de chave: o token leva `kid` no cabeçalho. Só a chave ATUAL assina; as anteriores
/// (`JWT_SECRET_PREVIOUS`) apenas validam, até os tokens antigos expirarem. Token sem `kid` ou
/// com `kid` desconhecido é rejeitado.
pub struct TokenService {
    enc: EncodingKey,
    kid: String,
    keys: Vec<(String, DecodingKey)>,
    validation: Validation,
    issuer: String,
    ttl: Duration,
}

impl TokenService {
    pub fn new(secret: &[u8], issuer: &str, ttl: Duration) -> Self {
        Self::with_previous(secret, &[], issuer, ttl)
    }

    pub fn with_previous(secret: &[u8], previous: &[&[u8]], issuer: &str, ttl: Duration) -> Self {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.algorithms = vec![Algorithm::HS256];
        validation.set_issuer(&[issuer]);
        validation.set_audience(&[AUDIENCE]);
        validation.set_required_spec_claims(&["exp", "iss", "aud", "sub", "nbf"]);
        validation.validate_nbf = true;
        validation.leeway = 15;
        let keys = std::iter::once(secret)
            .chain(previous.iter().copied())
            .map(|k| (key_id(k), DecodingKey::from_secret(k)))
            .collect();
        Self {
            enc: EncodingKey::from_secret(secret),
            kid: key_id(secret),
            keys,
            validation,
            issuer: issuer.to_string(),
            ttl,
        }
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    pub fn issue(&self, user_id: Uuid, client_id: &str, scope: &str) -> Result<String, jsonwebtoken::errors::Error> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let claims = Claims {
            iss: self.issuer.clone(),
            aud: AUDIENCE.to_string(),
            sub: user_id.to_string(),
            scope: scope.to_string(),
            client_id: client_id.to_string(),
            exp: now + self.ttl.as_secs(),
            iat: now,
            nbf: now,
            jti: Uuid::new_v4().to_string(),
        };
        let mut header = Header::new(Algorithm::HS256);
        header.kid = Some(self.kid.clone());
        encode(&header, &claims, &self.enc)
    }

    pub fn verify(&self, token: &str) -> Option<Claims> {
        let kid = decode_header(token).ok()?.kid?;
        let (_, key) = self.keys.iter().find(|(id, _)| *id == kid)?;
        decode::<Claims>(token, key, &self.validation).ok().map(|d| d.claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;

    fn svc() -> TokenService {
        TokenService::new(&[1u8; 32], "https://api.test", Duration::from_secs(600))
    }

    #[test]
    fn roundtrip() {
        let s = svc();
        let id = Uuid::new_v4();
        let t = s.issue(id, "cli", "ocr profile").unwrap();
        let c = s.verify(&t).expect("válido");
        assert_eq!(c.sub, id.to_string());
        assert!(c.has_scope("ocr") && !c.has_scope("ocr2") && !c.has_scope("oc"));
    }

    #[test]
    fn rejects_wrong_secret_issuer_and_expired() {
        let t = svc().issue(Uuid::new_v4(), "cli", "ocr").unwrap();
        assert!(TokenService::new(&[2u8; 32], "https://api.test", Duration::from_secs(600)).verify(&t).is_none());
        assert!(TokenService::new(&[1u8; 32], "https://outro", Duration::from_secs(600)).verify(&t).is_none());

        let mut s = svc();
        s.ttl = Duration::from_secs(0);
        s.validation.leeway = 0;
        let expired = s.issue(Uuid::new_v4(), "cli", "ocr").unwrap();
        std::thread::sleep(Duration::from_millis(1100));
        assert!(s.verify(&expired).is_none());
    }

    #[test]
    fn rejects_alg_none_and_tampering() {
        let s = svc();
        let valid = s.issue(Uuid::new_v4(), "cli", "ocr").unwrap();
        let parts: Vec<&str> = valid.split('.').collect();

        // alg=none, sem assinatura
        let header = B64.encode(br#"{"alg":"none","typ":"JWT"}"#);
        let none = format!("{header}.{}.", parts[1]);
        assert!(s.verify(&none).is_none());

        // payload alterado com a assinatura original
        let forged_payload = B64.encode(
            br#"{"iss":"https://api.test","aud":"api-ocr","sub":"x","scope":"admin","client_id":"c","exp":9999999999,"iat":0,"nbf":0,"jti":"j"}"#,
        );
        assert!(s.verify(&format!("{}.{forged_payload}.{}", parts[0], parts[2])).is_none());

        // lixo
        for junk in ["", "a.b.c", "....", "Bearer x"] {
            assert!(s.verify(junk).is_none());
        }
    }

    #[test]
    fn rotation_keeps_old_tokens_valid_until_the_old_key_is_removed() {
        let (old, new) = ([1u8; 32], [9u8; 32]);
        let before = TokenService::new(&old, "https://api.test", Duration::from_secs(600));
        let old_token = before.issue(Uuid::new_v4(), "cli", "ocr").unwrap();

        // Durante a rotação: assina com a nova, aceita as duas.
        let during = TokenService::with_previous(&new, &[&old], "https://api.test", Duration::from_secs(600));
        assert!(during.verify(&old_token).is_some(), "token antigo ainda vale");
        let new_token = during.issue(Uuid::new_v4(), "cli", "ocr").unwrap();
        assert!(during.verify(&new_token).is_some());
        assert_eq!(jsonwebtoken::decode_header(&new_token).unwrap().kid.as_deref(), Some(key_id(&new).as_str()));

        // Depois: a chave antiga saiu, o token antigo morre; a antiga nunca valida o novo.
        let after = TokenService::new(&new, "https://api.test", Duration::from_secs(600));
        assert!(after.verify(&old_token).is_none());
        assert!(after.verify(&new_token).is_some());
        assert!(before.verify(&new_token).is_none());
    }

    #[test]
    fn missing_or_unknown_kid_is_rejected() {
        let s = svc();
        let valid = s.issue(Uuid::new_v4(), "cli", "ocr").unwrap();
        let parts: Vec<&str> = valid.split('.').collect();
        // Mesmo payload, assinatura válida para um cabeçalho SEM kid.
        let no_kid = encode(&Header::new(Algorithm::HS256), &s.verify(&valid).unwrap(), &EncodingKey::from_secret(&[1u8; 32])).unwrap();
        assert!(s.verify(&no_kid).is_none(), "sem kid");
        // kid trocado: o cabeçalho muda, a assinatura deixa de bater e o kid não existe.
        let header = B64.encode(br#"{"alg":"HS256","typ":"JWT","kid":"0000000000000000"}"#);
        assert!(s.verify(&format!("{header}.{}.{}", parts[1], parts[2])).is_none());
    }
}
