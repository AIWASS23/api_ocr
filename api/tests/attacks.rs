mod common;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use common::*;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const SQLI: &[&str] = &[
    "' OR '1'='1",
    "' OR '1'='1' --",
    "admin'--",
    "'; DROP TABLE users; --",
    "\" OR \"\"=\"",
    "1; SELECT pg_sleep(5)--",
    "' UNION SELECT password_hash FROM users --",
    "x') OR ('1'='1",
    "\\'; DELETE FROM refresh_tokens; --",
    "%27%20OR%201=1",
];

async fn assert_db_intact(app: &TestApp) {
    let users: i64 = sqlx::query_scalar("SELECT count(*) FROM users").fetch_one(&app.db).await.unwrap();
    let clients: i64 = sqlx::query_scalar("SELECT count(*) FROM oauth_clients").fetch_one(&app.db).await.unwrap();
    assert_eq!((users, clients), (1, 1), "o banco foi alterado por um payload");
}

#[tokio::test]
async fn sql_injection_in_every_input_is_inert() {
    let app = TestApp::start().await;
    let (verifier, challenge) = pkce();
    let (csrf, _) = app.authorize_page(&challenge, "s").await;
    let started = std::time::Instant::now();

    for p in SQLI {
        // login (e-mail e senha)
        for (email, pass) in [(*p, PASSWORD), (EMAIL, *p), (*p, *p)] {
            let r = app.post_login(&csrf, &challenge, "s", email, pass).await;
            assert_ne!(r.status(), 303, "payload autenticou: {p:?}");
            assert_ne!(r.status(), 200);
        }
        // client_id / code / refresh_token no endpoint de token
        for (k, grant) in [("code", "authorization_code"), ("refresh_token", "refresh_token")] {
            let r = app
                .http
                .post(app.url("/oauth/token"))
                .form(&[("grant_type", grant), (k, p), ("client_id", CLIENT_ID), ("redirect_uri", REDIRECT), ("code_verifier", &verifier)])
                .send()
                .await
                .unwrap();
            assert!(r.status().is_client_error(), "{grant} {p:?} -> {}", r.status());
        }
        let r = app.http.post(app.url("/oauth/token")).form(&[("grant_type", "refresh_token"), ("client_id", p)]).send().await.unwrap();
        assert_eq!(r.status(), 401);
        // revoke, register
        let r = app.http.post(app.url("/oauth/revoke")).form(&[("token", *p), ("client_id", *p)]).send().await.unwrap();
        assert_eq!(r.status(), 200);
        let r = app.http.post(app.url("/auth/register")).json(&json!({ "email": p, "password": p })).send().await.unwrap();
        assert!(r.status().is_client_error(), "register {p:?}");
        // state / scope / redirect no authorize
        let mut q = app.authorize_query(&challenge, p);
        q.retain(|(n, _)| *n != "scope");
        q.push(("scope", p.to_string()));
        let r = app.http.get(app.url("/oauth/authorize")).query(&q).send().await.unwrap();
        assert_ne!(r.status(), 200);
        // Bearer
        let r = app.http.get(app.url("/v1/me")).bearer_auth(p).send().await.unwrap();
        assert_eq!(r.status(), 401);
    }

    assert!(started.elapsed() < Duration::from_secs(60), "pg_sleep não pode ter sido executado");
    assert_db_intact(&app).await;
    // As tentativas com o e-mail real bloquearam a conta para ESTA origem (comportamento esperado)...
    let locked: bool = sqlx::query_scalar("SELECT locked_until > now() FROM login_attempts WHERE origin = '127.0.0.1'")
        .fetch_one(&app.db)
        .await
        .unwrap();
    assert!(locked);
    // ...e, depois de liberada, o usuário legítimo continua entrando: nada foi corrompido.
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    api_ocr::store::unlock_user(&app.db, uid).await.unwrap();
    let _ = app.tokens().await;
}

#[tokio::test]
async fn forged_and_malformed_tokens_are_rejected() {
    let app = TestApp::start().await;
    let (access, _) = app.tokens().await;
    let parts: Vec<&str> = access.split('.').collect();

    let none_hdr = B64.encode(br#"{"alg":"none","typ":"JWT"}"#);
    let admin_payload = B64.encode(
        json!({"iss":ISSUER,"aud":"api-ocr","sub":uuid::Uuid::new_v4().to_string(),"scope":"ocr profile admin","client_id":CLIENT_ID,"exp":9999999999u64,"iat":0,"nbf":0,"jti":"x"}).to_string(),
    );
    let hs512_hdr = B64.encode(br#"{"alg":"HS512","typ":"JWT"}"#);

    let candidates = vec![
        format!("{none_hdr}.{}.", parts[1]),
        format!("{none_hdr}.{admin_payload}."),
        format!("{}.{admin_payload}.{}", parts[0], parts[2]), // payload trocado, assinatura antiga
        format!("{hs512_hdr}.{}.{}", parts[1], parts[2]),    // troca de algoritmo
        format!("{}.{}.AAAA", parts[0], parts[1]),
        format!("{}.{}", parts[0], parts[1]),
        "a.b.c".into(),
        "x".repeat(10_000),
        String::new(),
    ];
    for t in candidates {
        let r = app.http.get(app.url("/v1/me")).bearer_auth(&t).send().await.unwrap();
        assert_eq!(r.status(), 401, "token forjado aceito: {:.40}", t);
    }

    // Esquemas/formatos de cabeçalho inválidos.
    for h in ["Basic dXNlcjpwYXNz", "Bearer", "Bearer  ", &format!("Bearer {access} extra"), &format!("Token {access}")] {
        let r = app.http.get(app.url("/v1/me")).header("authorization", h).send().await.unwrap();
        assert_eq!(r.status(), 401, "{h:.30}");
    }
    // Dois cabeçalhos Authorization.
    let r = app.http.get(app.url("/v1/me")).header("authorization", format!("Bearer {access}")).header("authorization", "Bearer outro").send().await.unwrap();
    assert_eq!(r.status(), 401);

    // Token assinado com OUTRO segredo.
    let other = api_ocr::tokens::TokenService::new(b"outro-segredo-outro-segredo-0000000000", ISSUER, Duration::from_secs(60));
    let t = other.issue(uuid::Uuid::new_v4(), CLIENT_ID, "ocr").unwrap();
    assert_eq!(app.http.post(app.url("/v1/data_preparation")).bearer_auth(&t).send().await.unwrap().status(), 401);

    // Original continua válido.
    assert_eq!(app.http.get(app.url("/v1/me")).bearer_auth(&access).send().await.unwrap().status(), 200);
}

#[tokio::test]
async fn scope_is_enforced_on_ocr() {
    let app = TestApp::start().await;
    let (_, challenge) = pkce();
    // cliente que só tem `profile`
    api_ocr::store::create_client(&app.db, "so-profile", "SoProfile", &[REDIRECT.to_string()], &["profile".to_string()]).await.unwrap();
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    let t = app.state.tokens.issue(uid, "so-profile", "profile").unwrap();
    let form = reqwest::multipart::Form::new().part("file", reqwest::multipart::Part::bytes(minimal_docx("oi")).file_name("a.docx"));
    let r = app.http.post(app.url("/v1/data_preparation")).bearer_auth(&t).multipart(form).send().await.unwrap();
    assert_eq!(r.status(), 403);

    // E o cliente não consegue pedir um escopo que não possui.
    let mut q = app.authorize_query(&challenge, "s");
    q.retain(|(n, _)| !matches!(*n, "client_id" | "scope"));
    q.push(("client_id", "so-profile".into()));
    q.push(("scope", "ocr".into()));
    let r = app.http.get(app.url("/oauth/authorize")).query(&q).send().await.unwrap();
    assert_eq!(r.status(), 303);
    assert!(r.headers()["location"].to_str().unwrap().contains("error=invalid_scope"));
}

#[tokio::test]
async fn rate_limit_on_auth_endpoints_returns_429_with_retry_after() {
    let app = TestApp::start_with(|c| c.rate_auth_per_min = 10).await; // rajada = 5
    let mut got_429 = None;
    for _ in 0..15 {
        let r = app.http.post(app.url("/oauth/token")).form(&[("grant_type", "refresh_token"), ("client_id", CLIENT_ID), ("refresh_token", "x")]).send().await.unwrap();
        if r.status() == 429 {
            got_429 = Some(r);
            break;
        }
    }
    let r = got_429.expect("deveria limitar");
    assert!(r.headers().contains_key("retry-after"));
    assert_eq!(r.headers()["cache-control"], "no-store");
    // A rota protegida de outro grupo não é afetada pelo limite de auth.
    assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
}

#[tokio::test]
async fn general_rate_limit_applies_to_everything() {
    let app = TestApp::start_with(|c| c.rate_general_per_min = 30).await; // rajada = 10
    let mut limited = false;
    for _ in 0..40 {
        if app.http.get(app.url("/healthz")).send().await.unwrap().status() == 429 {
            limited = true;
            break;
        }
    }
    assert!(limited);
}

#[tokio::test]
async fn forged_forwarded_for_does_not_bypass_limits_without_trusted_proxy() {
    let app = TestApp::start_with(|c| c.rate_general_per_min = 30).await;
    let mut limited = false;
    for i in 0..40 {
        let r = app.http.get(app.url("/healthz")).header("x-forwarded-for", format!("10.0.0.{i}")).send().await.unwrap();
        if r.status() == 429 {
            limited = true;
            break;
        }
    }
    assert!(limited, "X-Forwarded-For não pode ser aceito sem TRUST_PROXY");
}

#[tokio::test]
async fn oversized_bodies_are_rejected() {
    let app = TestApp::start().await;
    let big = "a".repeat(200 * 1024);
    let r = app.http.post(app.url("/oauth/token")).form(&[("grant_type", "refresh_token"), ("refresh_token", big.as_str()), ("client_id", CLIENT_ID)]).send().await.unwrap();
    assert_eq!(r.status(), 413);
    let r = app.http.post(app.url("/auth/register")).json(&json!({"email": "a@b.com", "password": big})).send().await.unwrap();
    assert_eq!(r.status(), 413);

    // Upload de OCR acima do limite (2 MiB no teste).
    let (access, _) = app.tokens().await;
    let part = reqwest::multipart::Part::bytes(vec![b'%'; 3 * 1024 * 1024]).file_name("grande.pdf");
    let r = app.http.post(app.url("/v1/data_preparation")).bearer_auth(&access).multipart(reqwest::multipart::Form::new().part("file", part)).send().await.unwrap();
    assert_eq!(r.status(), 413);
}

#[tokio::test]
async fn malformed_json_and_unknown_fields_are_400_without_leaks() {
    let app = TestApp::start().await;
    let r = app.http.post(app.url("/auth/register")).header("content-type", "application/json").body("{not json").send().await.unwrap();
    assert!(r.status().is_client_error());
    let r = app.http.post(app.url("/auth/register")).json(&json!({"email":"a@b.com","password":"uma-senha-bem-longa","is_admin":true})).send().await.unwrap();
    assert!(r.status().is_client_error(), "campos desconhecidos devem ser recusados");
    let body = r.text().await.unwrap();
    assert!(!body.contains("src/") && !body.contains("panicked"));
}

#[tokio::test]
async fn registration_rules() {
    let app = TestApp::start().await;
    let ok = app.http.post(app.url("/auth/register")).json(&json!({"email":" Nova@Exemplo.com ","password":"outra-senha-bem-longa"})).send().await.unwrap();
    assert_eq!(ok.status(), 201);
    // E-mail repetido (mesmo com caixa diferente).
    let dup = app.http.post(app.url("/auth/register")).json(&json!({"email":"nova@exemplo.COM","password":"outra-senha-bem-longa"})).send().await.unwrap();
    assert_eq!(dup.status(), 409);
    // Senha curta / fraca.
    for pw in ["curta", "aaaaaaaaaaaaaaaa", "Password1234"] {
        let r = app.http.post(app.url("/auth/register")).json(&json!({"email":"x@exemplo.com","password":pw})).send().await.unwrap();
        assert_eq!(r.status(), 400, "{pw}");
    }
    // A senha nunca é guardada em claro.
    let hash: String = sqlx::query_scalar("SELECT password_hash FROM users WHERE email='nova@exemplo.com'").fetch_one(&app.db).await.unwrap();
    assert!(hash.starts_with("$argon2id$") && !hash.contains("outra-senha"));
}

#[tokio::test]
async fn registration_can_be_disabled() {
    let app = TestApp::start_with(|c| c.allow_registration = false).await;
    let r = app.http.post(app.url("/auth/register")).json(&json!({"email":"x@exemplo.com","password":"uma-senha-bem-longa"})).send().await.unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn unknown_routes_and_methods() {
    let app = TestApp::start().await;
    assert_eq!(app.http.get(app.url("/admin")).send().await.unwrap().status(), 404);
    assert_eq!(app.http.get(app.url("/../../etc/passwd")).send().await.unwrap().status(), 404);
    assert_eq!(app.http.delete(app.url("/v1/me")).send().await.unwrap().status(), 405);
    assert_eq!(app.http.get(app.url("/oauth/token")).send().await.unwrap().status(), 405);
}

#[tokio::test]
async fn cors_is_closed_by_default() {
    let app = TestApp::start().await;
    let r = app.http.get(app.url("/healthz")).header("origin", "https://evil.example").send().await.unwrap();
    assert!(!r.headers().contains_key("access-control-allow-origin"));

    let app = TestApp::start_with(|c| c.cors_origins = vec!["https://app.exemplo.com".into()]).await;
    let ok = app.http.get(app.url("/healthz")).header("origin", "https://app.exemplo.com").send().await.unwrap();
    assert_eq!(ok.headers()["access-control-allow-origin"], "https://app.exemplo.com");
    let bad = app.http.get(app.url("/healthz")).header("origin", "https://evil.example").send().await.unwrap();
    assert!(!bad.headers().contains_key("access-control-allow-origin"));
}

// ------------------------------------------------------------------ DoS de conexão

#[tokio::test]
async fn slowloris_connection_is_dropped() {
    let app = TestApp::start().await;
    let mut s = tokio::net::TcpStream::connect(app.addr).await.unwrap();
    // Cabeçalhos que nunca terminam.
    s.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\nX-Slow: ").await.unwrap();
    let mut buf = [0u8; 256];
    let res = tokio::time::timeout(Duration::from_secs(20), s.read(&mut buf)).await;
    match res {
        Ok(Ok(n)) => {
            let text = String::from_utf8_lossy(&buf[..n]);
            assert!(n == 0 || text.contains("408") || text.contains("400"), "resposta inesperada: {text}");
        }
        Ok(Err(_)) => {} // conexão resetada: ok
        Err(_) => panic!("conexão lenta não foi encerrada em 20 s"),
    }
}

#[tokio::test]
async fn per_ip_connection_cap() {
    let app = TestApp::start_with(|c| c.max_connections_per_ip = 5).await;
    let mut held = vec![];
    for _ in 0..5 {
        held.push(tokio::net::TcpStream::connect(app.addr).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    // A 6ª é fechada pelo servidor sem resposta.
    let mut extra = tokio::net::TcpStream::connect(app.addr).await.unwrap();
    let _ = extra.write_all(b"GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n").await;
    let mut buf = [0u8; 64];
    let n = tokio::time::timeout(Duration::from_secs(5), extra.read(&mut buf)).await.expect("deveria fechar logo");
    assert!(matches!(n, Ok(0) | Err(_)), "conexão excedente não deveria ser atendida");

    // Liberando uma vaga, volta a funcionar.
    drop(held.pop());
    tokio::time::sleep(Duration::from_millis(300)).await;
    let r = app.http.get(app.url("/healthz")).send().await;
    assert!(r.is_ok());
}

#[tokio::test]
async fn huge_headers_are_rejected() {
    let app = TestApp::start().await;
    let r = app.http.get(app.url("/healthz")).header("x-big", "a".repeat(64 * 1024)).send().await;
    // Conexão fechada (Err) também conta como rejeição.
    if let Ok(resp) = r {
        assert!(resp.status().is_client_error(), "{}", resp.status());
    }
    let r = app.http.get(app.url(&format!("/{}", "a".repeat(64 * 1024)))).send().await;
    if let Ok(resp) = r { assert!(resp.status().is_client_error()) }
}

#[allow(dead_code)]
fn _unused(_: Value) {}
