//! Validação de entrada. Tudo aqui é "lista de permitidos": o que não casa, rejeita.

pub const SUPPORTED_SCOPES: &[&str] = &["ocr", "profile"];

pub fn normalize_email(raw: &str) -> Option<String> {
    let e = raw.trim().to_lowercase();
    if e.len() < 3 || e.len() > 254 {
        return None;
    }
    if e.chars().any(|c| c.is_whitespace() || c.is_control() || matches!(c, '<' | '>' | '"' | '\'' | '\\' | ',' | ';' | '(' | ')')) {
        return None;
    }
    let (local, domain) = e.split_once('@')?;
    if local.is_empty() || local.len() > 64 || domain.contains('@') {
        return None;
    }
    if !domain.contains('.') || domain.starts_with('.') || domain.ends_with('.') || domain.contains("..") {
        return None;
    }
    Some(e)
}

const COMMON_PASSWORDS: &[&str] = &[
    "password1234", "123456789012", "qwertyuiop12", "senha1234567", "administrator",
    "passw0rd1234", "letmein12345", "welcome12345", "iloveyou1234", "abc123456789",
];

/// 12 a 128 caracteres (o teto evita DoS por senhas gigantes no Argon2).
pub fn check_password(password: &str, email: &str) -> Result<(), &'static str> {
    let n = password.chars().count();
    if n < 12 {
        return Err("a senha precisa ter ao menos 12 caracteres");
    }
    if n > 128 {
        return Err("a senha pode ter no máximo 128 caracteres");
    }
    if password.trim().is_empty() || password.chars().all(|c| c == password.chars().next().unwrap_or(' ')) {
        return Err("senha muito fraca");
    }
    let lower = password.to_lowercase();
    if lower == email || COMMON_PASSWORDS.contains(&lower.as_str()) {
        return Err("senha muito fraca");
    }
    Ok(())
}

fn valid_scope_token(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b':' | b'.' | b'_' | b'-'))
}

/// Normaliza uma lista de escopos separados por espaço (ordenada, sem repetição).
pub fn parse_scopes(raw: &str) -> Option<Vec<String>> {
    let mut v: Vec<String> = Vec::new();
    for s in raw.split_ascii_whitespace() {
        if !valid_scope_token(s) {
            return None;
        }
        if !v.iter().any(|x| x == s) {
            v.push(s.to_string());
        }
    }
    if v.len() > 10 {
        return None;
    }
    v.sort();
    Some(v)
}

/// Verificador PKCE (RFC 7636): 43–128 caracteres do alfabeto não reservado.
pub fn valid_code_verifier(v: &str) -> bool {
    (43..=128).contains(&v.len()) && v.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~'))
}

/// `code_challenge` S256 = base64url de 32 bytes = 43 caracteres.
pub fn valid_code_challenge(c: &str) -> bool {
    c.len() == 43 && c.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
}

/// Redirect URIs registráveis: https, ou http em loopback; sem fragmento nem credenciais.
pub fn valid_redirect_uri(raw: &str) -> bool {
    let Ok(u) = url::Url::parse(raw) else { return false };
    if u.fragment().is_some() || !u.username().is_empty() || u.password().is_some() || raw.len() > 2048 {
        return false;
    }
    match u.scheme() {
        "https" => u.host_str().is_some(),
        "http" => matches!(u.host_str(), Some("localhost" | "127.0.0.1" | "[::1]")),
        _ => false,
    }
}

pub fn valid_client_id(s: &str) -> bool {
    (3..=64).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emails() {
        assert_eq!(normalize_email("  Ana@Exemplo.COM "), Some("ana@exemplo.com".into()));
        // espaços/quebras NAS PONTAS são aparados; no meio (injeção de cabeçalho) são recusados
        assert_eq!(normalize_email("a@b.com\n"), Some("a@b.com".into()));
        for bad in ["", "a", "sem-arroba", "@x.com", "a@b", "a@@b.com", "a b@c.com", "a@b..com", "a@.com", "<x>@y.com", "a@b.com\nBcc: x@y.com", "a\"@b.com"] {
            assert_eq!(normalize_email(bad), None, "{bad:?}");
        }
        assert_eq!(normalize_email(&format!("{}@x.com", "a".repeat(250))), None);
    }

    #[test]
    fn passwords() {
        assert!(check_password("uma-senha-boa-123", "a@b.com").is_ok());
        assert!(check_password("curta", "a@b.com").is_err());
        assert!(check_password(&"x".repeat(129), "a@b.com").is_err());
        assert!(check_password("aaaaaaaaaaaaaaaa", "a@b.com").is_err());
        assert!(check_password("Password1234", "a@b.com").is_err());
    }

    #[test]
    fn scopes_and_pkce() {
        assert_eq!(parse_scopes("profile ocr ocr"), Some(vec!["ocr".into(), "profile".into()]));
        assert_eq!(parse_scopes("oc\"r"), None);
        assert_eq!(parse_scopes(&"a ".repeat(11)), Some(vec!["a".into()]));
        assert!(valid_code_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"));
        assert!(!valid_code_verifier("curto"));
        assert!(!valid_code_verifier(&"a$".repeat(30)));
        assert!(valid_code_challenge("E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"));
        assert!(!valid_code_challenge("E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-c="));
    }

    #[test]
    fn redirect_uris() {
        assert!(valid_redirect_uri("https://app.exemplo.com/callback"));
        assert!(valid_redirect_uri("http://localhost:3000/cb"));
        assert!(valid_redirect_uri("http://127.0.0.1:8123/cb"));
        for bad in ["http://evil.com/cb", "javascript:alert(1)", "https://a.com/cb#frag", "https://u:p@a.com/", "ftp://a.com", "//a.com", "", "/relativo"] {
            assert!(!valid_redirect_uri(bad), "{bad}");
        }
    }
}
