mod common;
use common::*;
use serde_json::Value;

#[tokio::test]
async fn full_flow_authorize_token_protected_refresh_revoke() {
    let app = TestApp::start().await;

    // Rota protegida sem token.
    let r = app.http.get(app.url("/v1/me")).send().await.unwrap();
    assert_eq!(r.status(), 401);
    assert!(r.headers()["www-authenticate"].to_str().unwrap().starts_with("Bearer"));

    let (access, refresh) = app.tokens().await;

    let me: Value = app.http.get(app.url("/v1/me")).bearer_auth(&access).send().await.unwrap().json().await.unwrap();
    assert_eq!(me["email"], EMAIL);

    // Refresh rotaciona.
    let r = app.refresh(&refresh).await;
    assert_eq!(r.status(), 200);
    let j: Value = r.json().await.unwrap();
    let refresh2 = j["refresh_token"].as_str().unwrap().to_string();
    assert_ne!(refresh2, refresh);
    assert!(j["access_token"].as_str().unwrap().split('.').count() == 3);

    // Reutilizar o refresh antigo => invalid_grant E derruba a família inteira.
    let r = app.refresh(&refresh).await;
    assert_eq!(r.status(), 400);
    assert_eq!(r.json::<Value>().await.unwrap()["error"], "invalid_grant");
    let r = app.refresh(&refresh2).await;
    assert_eq!(r.status(), 400, "o token novo também deve estar revogado após o reuso");

    // Revogação explícita.
    let (_, refresh3) = app.tokens().await;
    let r = app.http.post(app.url("/oauth/revoke")).form(&[("token", refresh3.as_str()), ("client_id", CLIENT_ID)]).send().await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(app.refresh(&refresh3).await.status(), 400);

    // Revogar lixo também responde 200 (RFC 7009).
    let r = app.http.post(app.url("/oauth/revoke")).form(&[("token", "lixo"), ("client_id", CLIENT_ID)]).send().await.unwrap();
    assert_eq!(r.status(), 200);
}

#[tokio::test]
async fn token_responses_are_not_cacheable_and_have_security_headers() {
    let app = TestApp::start().await;
    let (verifier, challenge) = pkce();
    let code = app.login_for_code(&challenge, "s").await;
    let r = app.exchange(&code, &verifier).await;
    assert_eq!(r.headers()["cache-control"], "no-store");
    assert_eq!(r.headers()["pragma"], "no-cache");
    for (k, v) in [("x-content-type-options", "nosniff"), ("x-frame-options", "DENY"), ("referrer-policy", "no-referrer")] {
        assert_eq!(r.headers()[k], v, "{k}");
    }
    assert!(r.headers().contains_key("content-security-policy"));
    assert!(r.headers().contains_key("x-request-id"));
}

#[tokio::test]
async fn authorization_code_cannot_be_replayed_and_replay_revokes_tokens() {
    let app = TestApp::start().await;
    let (verifier, challenge) = pkce();
    let code = app.login_for_code(&challenge, "s").await;

    let first = app.exchange(&code, &verifier).await;
    assert_eq!(first.status(), 200);
    let refresh = first.json::<Value>().await.unwrap()["refresh_token"].as_str().unwrap().to_string();

    let second = app.exchange(&code, &verifier).await;
    assert_eq!(second.status(), 400);
    assert_eq!(second.json::<Value>().await.unwrap()["error"], "invalid_grant");

    // O replay derruba os tokens que o código já havia gerado.
    assert_eq!(app.refresh(&refresh).await.status(), 400);
}

#[tokio::test]
async fn pkce_is_enforced() {
    let app = TestApp::start().await;
    let (_, challenge) = pkce();
    let code = app.login_for_code(&challenge, "s").await;
    let (other_verifier, _) = pkce();
    let r = app.exchange(&code, &other_verifier).await;
    assert_eq!(r.status(), 400);
    assert_eq!(r.json::<Value>().await.unwrap()["error"], "invalid_grant");

    // O código foi queimado pela tentativa errada.
    let (verifier, challenge) = pkce();
    let code = app.login_for_code(&challenge, "s").await;
    assert_eq!(app.exchange(&code, &verifier).await.status(), 200);

    // Pedido de autorização sem PKCE / com método plain é recusado (redirect com erro).
    for (k, v) in [("code_challenge_method", "plain")] {
        let mut q = app.authorize_query(&challenge, "s");
        q.retain(|(n, _)| *n != k);
        q.push((k, v.to_string()));
        let r = app.http.get(app.url("/oauth/authorize")).query(&q).send().await.unwrap();
        assert_eq!(r.status(), 303);
        assert!(r.headers()["location"].to_str().unwrap().contains("error=invalid_request"));
    }
    let mut q = app.authorize_query(&challenge, "s");
    q.retain(|(n, _)| *n != "code_challenge");
    let r = app.http.get(app.url("/oauth/authorize")).query(&q).send().await.unwrap();
    assert_eq!(r.status(), 303);
}

#[tokio::test]
async fn invalid_client_or_redirect_never_redirects() {
    let app = TestApp::start().await;
    let (_, challenge) = pkce();
    for (field, value) in [
        ("redirect_uri", "http://evil.example/steal"),
        ("redirect_uri", "http://localhost:3000/callback/../x"),
        ("redirect_uri", "http://localhost:3000/callback?x=1"),
        ("redirect_uri", "http://localhost:3000/CALLBACK"),
        ("client_id", "nao-existe"),
        ("client_id", "x' OR '1'='1"),
    ] {
        let mut q = app.authorize_query(&challenge, "s");
        q.retain(|(n, _)| *n != field);
        q.push((field, value.to_string()));
        let r = app.http.get(app.url("/oauth/authorize")).query(&q).send().await.unwrap();
        assert_eq!(r.status(), 400, "{field}={value}");
        assert!(!r.headers().contains_key("location"), "não pode redirecionar: {field}={value}");
    }
}

#[tokio::test]
async fn login_page_is_hardened_and_escapes_input() {
    let app = TestApp::start().await;
    let (_, challenge) = pkce();
    let evil_state = "\"><script>alert(1)</script>";
    let (_, html) = app.authorize_page(&challenge, evil_state).await;
    assert!(!html.contains("<script>alert(1)"), "state precisa sair escapado");
    assert!(html.contains("&quot;&gt;&lt;script&gt;"));

    let r = app.http.get(app.url("/oauth/authorize")).query(&app.authorize_query(&challenge, "s")).send().await.unwrap();
    let csp = r.headers()["content-security-policy"].to_str().unwrap().to_string();
    assert!(csp.contains("frame-ancestors 'none'") && csp.contains("default-src 'none'") && !csp.contains("unsafe-inline"));
    assert_eq!(r.headers()["x-frame-options"], "DENY");
    let cookie = r.headers()["set-cookie"].to_str().unwrap().to_ascii_lowercase();
    assert!(cookie.contains("httponly") && cookie.contains("samesite=strict"));
}

#[tokio::test]
async fn login_requires_csrf() {
    let app = TestApp::start().await;
    let (_, challenge) = pkce();
    let (csrf, _) = app.authorize_page(&challenge, "s").await;

    // Sem cookie.
    let mut form = app.authorize_query(&challenge, "s");
    form.extend([("email", EMAIL.to_string()), ("password", PASSWORD.to_string()), ("csrf", csrf.clone())]);
    let r = app.http.post(app.url("/oauth/authorize")).form(&form).send().await.unwrap();
    assert_eq!(r.status(), 403);

    // Cookie diferente do campo.
    let r = app.http.post(app.url("/oauth/authorize")).header("cookie", "csrf=outro-valor").form(&form).send().await.unwrap();
    assert_eq!(r.status(), 403);
}

#[tokio::test]
async fn wrong_password_is_generic_and_locks_account_after_5_failures() {
    let app = TestApp::start().await;
    let (_, challenge) = pkce();
    let (csrf, _) = app.authorize_page(&challenge, "s").await;

    let wrong_known = app.post_login(&csrf, &challenge, "s", EMAIL, "senha-errada-000").await;
    let wrong_unknown = app.post_login(&csrf, &challenge, "s", "ninguem@exemplo.com", "senha-errada-000").await;
    assert_eq!(wrong_known.status(), 401);
    assert_eq!(wrong_unknown.status(), 401);
    let (a, b) = (wrong_known.text().await.unwrap(), wrong_unknown.text().await.unwrap());
    assert_eq!(a, b, "a resposta não pode distinguir usuário existente de inexistente");

    for _ in 0..4 {
        app.post_login(&csrf, &challenge, "s", EMAIL, "senha-errada-000").await;
    }
    // Agora a conta está bloqueada: nem a senha CERTA entra.
    let r = app.post_login(&csrf, &challenge, "s", EMAIL, PASSWORD).await;
    assert_eq!(r.status(), 401, "conta bloqueada deve recusar até a senha correta");
}

#[tokio::test]
async fn refresh_scope_can_only_narrow() {
    let app = TestApp::start().await;
    let (_, refresh) = app.tokens().await;
    let r = app
        .http
        .post(app.url("/oauth/token"))
        .form(&[("grant_type", "refresh_token"), ("refresh_token", refresh.as_str()), ("client_id", CLIENT_ID), ("scope", "ocr admin")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    assert_eq!(r.json::<Value>().await.unwrap()["error"], "invalid_scope");
}

#[tokio::test]
async fn refresh_token_is_bound_to_its_client() {
    let app = TestApp::start().await;
    api_ocr::store::create_client(&app.db, "outro-cliente", "Outro", &[REDIRECT.to_string()], &["ocr".to_string()]).await.unwrap();
    let (_, refresh) = app.tokens().await;
    let r = app
        .http
        .post(app.url("/oauth/token"))
        .form(&[("grant_type", "refresh_token"), ("refresh_token", refresh.as_str()), ("client_id", "outro-cliente")])
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    // Apresentá-lo por outro cliente é suspeito: o original também morre.
    assert_eq!(app.refresh(&refresh).await.status(), 400);
}

#[tokio::test]
async fn expired_refresh_family_is_rejected() {
    let app = TestApp::start().await;
    let (_, refresh) = app.tokens().await;
    sqlx::query("UPDATE refresh_tokens SET family_started_at = now() - interval '30 days'").execute(&app.db).await.unwrap();
    assert_eq!(app.refresh(&refresh).await.status(), 400, "vida máxima da família (7 dias no teste)");

    let (_, refresh) = app.tokens().await;
    sqlx::query("UPDATE refresh_tokens SET expires_at = now() - interval '1 minute' WHERE used_at IS NULL").execute(&app.db).await.unwrap();
    assert_eq!(app.refresh(&refresh).await.status(), 400);
}

#[tokio::test]
async fn unsupported_grants_are_rejected() {
    let app = TestApp::start().await;
    for g in ["password", "client_credentials", "implicit", ""] {
        let r = app.http.post(app.url("/oauth/token")).form(&[("grant_type", g), ("client_id", CLIENT_ID)]).send().await.unwrap();
        assert_eq!(r.status(), 400, "{g}");
    }
    let r = app.http.post(app.url("/oauth/token")).form(&[("grant_type", "authorization_code")]).send().await.unwrap();
    assert_eq!(r.status(), 401); // sem client_id
}

#[tokio::test]
async fn metadata_document() {
    let app = TestApp::start().await;
    let j: Value = app.http.get(app.url("/.well-known/oauth-authorization-server")).send().await.unwrap().json().await.unwrap();
    assert_eq!(j["code_challenge_methods_supported"][0], "S256");
    assert_eq!(j["grant_types_supported"].as_array().unwrap().len(), 2);
}
