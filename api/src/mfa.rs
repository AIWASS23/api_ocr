//! Segundo fator (TOTP, RFC 6238: 6 dígitos, passo de 30 s, HMAC-SHA1 — o que qualquer app
//! autenticador entende).
//!
//! O segredo fica cifrado no banco (ChaCha20-Poly1305) com uma chave derivada do
//! `PASSWORD_PEPPER`, e o id do usuário entra como dado associado: um vazamento só do banco não
//! revela os segredos, e não dá para copiar o segredo de um usuário para outro. Cada código vale
//! uma única vez (`totp_last_step`).
//!
//! Ativação e desativação são feitas pelo operador (`api_ocr mfa-enable/mfa-disable`), como a
//! criação de usuários: um access token roubado não serve para trocar o segundo fator.

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use hmac::{Hmac, Mac};
use sha1::Sha1;
use sha2::Sha256;
use uuid::Uuid;

pub const STEP_SECS: u64 = 30;
const DIGITS: u32 = 6;
const SECRET_BYTES: usize = 20;
const NONCE_BYTES: usize = 12;

/// Cifra/decifra segredos TOTP.
pub struct MfaCipher {
    cipher: ChaCha20Poly1305,
}

impl MfaCipher {
    /// Chave = HMAC-SHA256(pepper, rótulo): separação de domínio em relação ao uso do pepper no Argon2.
    pub fn from_pepper(pepper: &[u8]) -> Self {
        let mut mac = <Hmac<Sha256> as hmac::KeyInit>::new_from_slice(pepper).expect("HMAC aceita qualquer tamanho de chave");
        mac.update(b"api_ocr totp-secret-encryption v1");
        let key = mac.finalize().into_bytes();
        Self { cipher: ChaCha20Poly1305::new_from_slice(&key).expect("chave de 32 bytes") }
    }

    pub fn encrypt(&self, user_id: Uuid, secret: &[u8]) -> Vec<u8> {
        let mut nonce = [0u8; NONCE_BYTES];
        rand::fill(&mut nonce[..]);
        let ct = self
            .cipher
            .encrypt(&Nonce::from(nonce), Payload { msg: secret, aad: user_id.as_bytes() })
            .expect("cifrar em memória não falha");
        [nonce.as_slice(), &ct].concat()
    }

    pub fn decrypt(&self, user_id: Uuid, blob: &[u8]) -> Option<Vec<u8>> {
        if blob.len() <= NONCE_BYTES {
            return None;
        }
        let (nonce, ct) = blob.split_at(NONCE_BYTES);
        let nonce: [u8; NONCE_BYTES] = nonce.try_into().ok()?;
        self.cipher.decrypt(&Nonce::from(nonce), Payload { msg: ct, aad: user_id.as_bytes() }).ok()
    }
}

pub fn new_secret() -> Vec<u8> {
    let mut s = vec![0u8; SECRET_BYTES];
    rand::fill(&mut s[..]);
    s
}

/// Código TOTP do passo `step` (RFC 4226 §5.3, truncamento dinâmico).
pub fn code_at(secret: &[u8], step: u64) -> u32 {
    let mut mac = <Hmac<Sha1> as hmac::KeyInit>::new_from_slice(secret).expect("HMAC aceita qualquer tamanho de chave");
    mac.update(&step.to_be_bytes());
    let h = mac.finalize().into_bytes();
    let off = (h[h.len() - 1] & 0x0f) as usize;
    let bin = u32::from_be_bytes([h[off] & 0x7f, h[off + 1], h[off + 2], h[off + 3]]);
    bin % 10u32.pow(DIGITS)
}

pub fn current_step(unix_secs: u64) -> u64 {
    unix_secs / STEP_SECS
}

/// Confere `code` contra o passo atual e os vizinhos (tolerância de relógio de ±30 s).
/// Devolve o passo que bateu (para registrar e impedir reuso).
pub fn verify(secret: &[u8], code: &str, now_step: u64) -> Option<u64> {
    let code = code.trim();
    if code.len() != DIGITS as usize || !code.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let wanted: u32 = code.parse().ok()?;
    let mut hit = None;
    // Sem retorno antecipado: o tempo não depende de qual passo bateu.
    for step in [now_step.saturating_sub(1), now_step, now_step + 1] {
        if crate::crypto::ct_eq(&code_at(secret, step).to_be_bytes(), &wanted.to_be_bytes()) {
            hit = Some(step);
        }
    }
    hit
}

/// Base32 (RFC 4648) sem padding, como os apps autenticadores esperam.
pub fn base32(data: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut out = String::new();
    let (mut buf, mut bits) = (0u32, 0u32);
    for &b in data {
        buf = (buf << 8) | b as u32;
        bits += 8;
        while bits >= 5 {
            out.push(ALPHABET[((buf >> (bits - 5)) & 31) as usize] as char);
            bits -= 5;
        }
    }
    if bits > 0 {
        out.push(ALPHABET[((buf << (5 - bits)) & 31) as usize] as char);
    }
    out
}

/// URI `otpauth://` para gerar o QR code no app autenticador.
pub fn otpauth_uri(issuer: &str, email: &str, secret: &[u8]) -> String {
    let label = url::form_urlencoded::byte_serialize(format!("{issuer}:{email}").as_bytes()).collect::<String>().replace('+', "%20");
    let mut u = url::Url::parse(&format!("otpauth://totp/{label}")).expect("URI fixa");
    u.query_pairs_mut()
        .append_pair("secret", &base32(secret))
        .append_pair("issuer", issuer)
        .append_pair("algorithm", "SHA1")
        .append_pair("digits", &DIGITS.to_string())
        .append_pair("period", &STEP_SECS.to_string());
    u.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc6238_vectors() {
        // Apêndice B da RFC 6238 (SHA-1), últimos 6 dígitos dos códigos de 8.
        let secret = b"12345678901234567890";
        for (t, code8) in [(59u64, 94287082u32), (1111111109, 7081804), (1234567890, 89005924), (2000000000, 69279037)] {
            assert_eq!(code_at(secret, current_step(t)), code8 % 1_000_000, "t={t}");
        }
    }

    #[test]
    fn verify_accepts_neighbours_and_rejects_garbage() {
        let s = new_secret();
        let step = 1_000_000;
        for (offset, ok) in [(-1i64, true), (0, true), (1, true), (2, false), (-2, false)] {
            let code = format!("{:06}", code_at(&s, (step as i64 + offset) as u64));
            assert_eq!(verify(&s, &code, step).is_some(), ok, "offset {offset}");
        }
        for junk in ["", "12345", "1234567", "abcdef", "12 345"] {
            assert!(verify(&s, junk, step).is_none(), "{junk:?}");
        }
    }

    #[test]
    fn secret_is_bound_to_the_user_and_the_key() {
        let c = MfaCipher::from_pepper(b"pepper-pepper-pepper-pepper-pepper!!");
        let (u1, u2) = (Uuid::new_v4(), Uuid::new_v4());
        let blob = c.encrypt(u1, b"segredo");
        assert_eq!(c.decrypt(u1, &blob).as_deref(), Some(&b"segredo"[..]));
        assert!(c.decrypt(u2, &blob).is_none(), "copiar para outro usuário não funciona");
        assert!(MfaCipher::from_pepper(b"outro-pepper-outro-pepper-outro-pepper").decrypt(u1, &blob).is_none());
        let mut tampered = blob.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert!(c.decrypt(u1, &tampered).is_none());
    }

    #[test]
    fn base32_rfc4648() {
        assert_eq!(base32(b""), "");
        assert_eq!(base32(b"f"), "MY");
        assert_eq!(base32(b"foobar"), "MZXW6YTBOI");
        assert!(otpauth_uri("API OCR", "a@b.com", b"12345678901234567890").starts_with("otpauth://totp/API%20OCR%3Aa%40b.com?secret=GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ&issuer="));
    }
}
