//! Bloqueio de login por origem, segundo fator (TOTP), revogação imediata de access tokens,
//! rotação do segredo JWT e alertas de segurança.

mod common;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use api_ocr::{mfa, store, tokens::TokenService};
use common::*;
use serde_json::Value;
use uuid::Uuid;

/// Login completo vindo de `ip` (via X-Forwarded-For; o app precisa de `trust_proxy`).
async fn login_from(app: &TestApp, ip: &str, password: &str, otp: Option<&str>) -> reqwest::Response {
    let (_, challenge) = pkce();
    let resp = app
        .http
        .get(app.url("/oauth/authorize"))
        .header("x-forwarded-for", ip)
        .query(&app.authorize_query(&challenge, "s"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let cookie = resp.headers()["set-cookie"].to_str().unwrap().to_string();
    let csrf = cookie.split(';').next().unwrap().split_once('=').unwrap().1.to_string();
    let mut form = app.authorize_query(&challenge, "s");
    form.push(("email", EMAIL.into()));
    form.push(("password", password.into()));
    form.push(("csrf", csrf.clone()));
    if let Some(code) = otp {
        form.push(("otp", code.into()));
    }
    app.http
        .post(app.url("/oauth/authorize"))
        .header("x-forwarded-for", ip)
        .header("cookie", format!("csrf={csrf}"))
        .form(&form)
        .send()
        .await
        .unwrap()
}

async fn user_id(app: &TestApp) -> Uuid {
    sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap()
}

async fn events(app: &TestApp, kind: &str) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM audit_log WHERE kind = $1").bind(kind).fetch_one(&app.db).await.unwrap()
}

async fn me(app: &TestApp, token: &str) -> u16 {
    app.http.get(app.url("/v1/me")).bearer_auth(token).send().await.unwrap().status().as_u16()
}

// ------------------------------------------------------------------ bloqueio

#[tokio::test]
async fn attacker_on_one_ip_cannot_lock_the_user_out_from_another() {
    let app = TestApp::start_with(|c| c.trust_proxy = true).await;
    for _ in 0..8 {
        assert_eq!(login_from(&app, "198.51.100.66", "senha-errada-000", None).await.status(), 401);
    }
    // O atacante está bloqueado até com a senha certa...
    assert_eq!(login_from(&app, "198.51.100.66", PASSWORD, None).await.status(), 401);
    // ...mas a vítima, de outro lugar, entra normalmente.
    assert_eq!(login_from(&app, "203.0.113.10", PASSWORD, None).await.status(), 303);
    assert!(events(&app, "login_ip_locked").await >= 1);
}

#[tokio::test]
async fn ipv6_attacker_cannot_dodge_the_lock_by_rotating_addresses_in_its_64() {
    let app = TestApp::start_with(|c| c.trust_proxy = true).await;
    for i in 1..=5 {
        login_from(&app, &format!("2001:db8:aa:bb::{i:x}"), "senha-errada-000", None).await;
    }
    assert_eq!(login_from(&app, "2001:db8:aa:bb::ffff", PASSWORD, None).await.status(), 401);
    assert_eq!(login_from(&app, "2001:db8:aa:cc::1", PASSWORD, None).await.status(), 303, "outro /64 não é afetado");
}

#[tokio::test]
async fn distributed_attack_locks_the_account_for_unknown_origins_only() {
    let app = TestApp::start_with(|c| c.trust_proxy = true).await;
    // O usuário já entrou deste IP: origem conhecida.
    assert_eq!(login_from(&app, "203.0.113.10", PASSWORD, None).await.status(), 303);

    // Ataque de muitas origens (simulado: contador da conta quase no limite).
    let uid = user_id(&app).await;
    sqlx::query("UPDATE users SET failed_attempts = $2 - 1, last_failed_at = now() WHERE id = $1")
        .bind(uid)
        .bind(store::ACCOUNT_MAX_FAILED_LOGINS)
        .execute(&app.db)
        .await
        .unwrap();
    assert_eq!(login_from(&app, "192.0.2.77", "senha-errada-000", None).await.status(), 401);
    assert_eq!(events(&app, "account_locked").await, 1);
    assert_eq!(events(&app, "security_alert").await, 1, "bloqueio da conta inteira deve alertar");

    // Origem desconhecida: barrada mesmo com a senha certa. Origem conhecida: entra.
    assert_eq!(login_from(&app, "192.0.2.200", PASSWORD, None).await.status(), 401);
    assert_eq!(login_from(&app, "203.0.113.10", PASSWORD, None).await.status(), 303);
}

#[tokio::test]
async fn blocked_and_wrong_password_look_the_same() {
    let app = TestApp::start_with(|c| c.trust_proxy = true).await;
    let wrong = login_from(&app, "198.51.100.1", "senha-errada-000", None).await;
    assert_eq!(wrong.status(), 401);
    let wrong = wrong.text().await.unwrap();
    for _ in 0..5 {
        login_from(&app, "198.51.100.1", "senha-errada-000", None).await;
    }
    let blocked = login_from(&app, "198.51.100.1", PASSWORD, None).await;
    assert_eq!(blocked.status(), 401);
    let blocked = blocked.text().await.unwrap();
    let message = |s: &str| s.split("class=\"err\"").nth(1).unwrap().split("</p>").next().unwrap().to_string();
    assert_eq!(message(&wrong), message(&blocked), "a mensagem não pode revelar o bloqueio");
}

#[tokio::test]
async fn repeated_failures_from_one_ip_raise_a_security_alert() {
    let app = TestApp::start_with(|c| {
        c.trust_proxy = true;
        c.security_alert_login_failures = 3;
    })
    .await;
    for _ in 0..4 {
        login_from(&app, "198.51.100.9", "senha-errada-000", None).await;
    }
    assert_eq!(events(&app, "security_alert").await, 1, "um alerta por janela, não um por falha");
    let detail: String = sqlx::query_scalar("SELECT detail FROM audit_log WHERE kind = 'security_alert'").fetch_one(&app.db).await.unwrap();
    assert!(detail.contains("3 falhas"), "{detail}");
}

// ------------------------------------------------------------------ MFA

fn now_step() -> u64 {
    mfa::current_step(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())
}

async fn enable_mfa(app: &TestApp) -> Vec<u8> {
    let uid = user_id(app).await;
    let secret = mfa::new_secret();
    store::set_totp_secret(&app.db, uid, Some(&app.state.mfa.encrypt(uid, &secret))).await.unwrap();
    secret
}

#[tokio::test]
async fn mfa_requires_a_valid_code_once() {
    let app = TestApp::start_with(|c| c.trust_proxy = true).await;
    let secret = enable_mfa(&app).await;
    let ip = "203.0.113.20";

    // Senha certa sem código, ou com código errado: recusado.
    assert_eq!(login_from(&app, ip, PASSWORD, None).await.status(), 401);
    let wrong = format!("{:06}", (mfa::code_at(&secret, now_step()) + 1) % 1_000_000);
    assert_eq!(login_from(&app, ip, PASSWORD, Some(&wrong)).await.status(), 401);
    assert!(events(&app, "mfa_failed").await >= 2);

    // Código certo: entra. O MESMO código de novo: recusado (não pode ser reaproveitado).
    let code = format!("{:06}", mfa::code_at(&secret, now_step()));
    assert_eq!(login_from(&app, ip, PASSWORD, Some(&code)).await.status(), 303);
    assert_eq!(login_from(&app, ip, PASSWORD, Some(&code)).await.status(), 401);
}

#[tokio::test]
async fn mfa_code_never_rescues_a_wrong_password() {
    let app = TestApp::start_with(|c| c.trust_proxy = true).await;
    let secret = enable_mfa(&app).await;
    let code = format!("{:06}", mfa::code_at(&secret, now_step()));
    assert_eq!(login_from(&app, "203.0.113.21", "senha-errada-000", Some(&code)).await.status(), 401);
    // O código não foi gasto por uma tentativa com senha errada.
    assert_eq!(login_from(&app, "203.0.113.21", PASSWORD, Some(&code)).await.status(), 303);
}

#[tokio::test]
async fn login_page_has_the_otp_field_for_everyone() {
    let app = TestApp::start().await;
    let (_, challenge) = pkce();
    let (_, html) = app.authorize_page(&challenge, "s").await;
    assert!(html.contains("name=\"otp\"") && html.contains("autocomplete=\"one-time-code\""));
}

// ------------------------------------------------------------------ revogação

#[tokio::test]
async fn refresh_token_theft_kills_the_access_tokens_immediately() {
    let app = TestApp::start().await;
    let (access, refresh) = app.tokens().await;
    assert_eq!(me(&app, &access).await, 200);

    // Uso legítimo, depois o atacante reapresenta o refresh antigo.
    assert_eq!(app.refresh(&refresh).await.status(), 200);
    tokio::time::sleep(Duration::from_millis(1100)).await; // iat é em segundos
    assert_eq!(app.refresh(&refresh).await.status(), 400);

    // O access token, que valeria mais 10 minutos, morre na hora.
    assert_eq!(me(&app, &access).await, 401);
    assert_eq!(events(&app, "refresh_reuse_detected").await, 1);
    assert_eq!(events(&app, "security_alert").await, 1);

    // Um login novo funciona normalmente.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let (fresh, _) = app.tokens().await;
    assert_eq!(me(&app, &fresh).await, 200);
}

#[tokio::test]
async fn revoke_all_sessions_and_deactivation_take_effect_at_once() {
    let app = TestApp::start().await;
    let uid = user_id(&app).await;
    let (access, refresh) = app.tokens().await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    store::revoke_all_sessions(&app.db, uid).await.unwrap();
    assert_eq!(me(&app, &access).await, 401);
    assert_eq!(app.refresh(&refresh).await.status(), 400);

    tokio::time::sleep(Duration::from_millis(1100)).await;
    let (access, _) = app.tokens().await;
    assert_eq!(me(&app, &access).await, 200);
    sqlx::query("UPDATE users SET is_active = false WHERE id = $1").bind(uid).execute(&app.db).await.unwrap();
    assert_eq!(me(&app, &access).await, 401, "usuário desativado perde o acesso na hora");
}

// ------------------------------------------------------------------ rotação do segredo JWT

#[tokio::test]
async fn jwt_secret_rotation_accepts_old_tokens_only_while_configured() {
    let old = b"segredo-antigo-segredo-antigo-segredo-antigo".to_vec();
    let app = TestApp::start_with(|c| c.jwt_previous_secrets = vec![old.clone()]).await;
    let uid = user_id(&app).await;
    let issued_by_old = TokenService::new(&old, ISSUER, Duration::from_secs(600)).issue(uid, CLIENT_ID, "ocr profile").unwrap();
    assert_eq!(me(&app, &issued_by_old).await, 200, "token da chave anterior ainda vale na rotação");

    let (current, _) = app.tokens().await;
    let body: Value = app.http.get(app.url("/v1/me")).bearer_auth(&current).send().await.unwrap().json().await.unwrap();
    assert_eq!(body["id"], uid.to_string());

    let app2 = TestApp::start().await; // sem JWT_SECRET_PREVIOUS
    let uid2 = user_id(&app2).await;
    let t = TokenService::new(&old, ISSUER, Duration::from_secs(600)).issue(uid2, CLIENT_ID, "ocr profile").unwrap();
    assert_eq!(me(&app2, &t).await, 401, "fim da rotação: a chave antiga não vale mais");
}
