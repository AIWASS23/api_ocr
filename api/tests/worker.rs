//! O worker de OCR é um processo separado: travar, morrer ou responder lixo não derruba a API,
//! não vaza nada para o cliente e gera evento de segurança. Os casos patológicos usam um
//! "worker" de mentira (script) no lugar do binário.

mod common;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use common::*;
use reqwest::multipart::{Form, Part};

async fn upload(app: &TestApp, token: &str) -> reqwest::Response {
    let form = Form::new().part("file", Part::bytes(minimal_docx("texto")).file_name("a.docx"));
    app.http.post(app.url("/v1/ocr")).bearer_auth(token).multipart(form).send().await.unwrap()
}

/// Grava um script executável que faz o papel do worker.
fn fake_worker(dir: &Path, body: &str) -> PathBuf {
    let p = dir.join("worker.sh");
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

async fn events(app: &TestApp, kind: &str) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM audit_log WHERE kind = $1").bind(kind).fetch_one(&app.db).await.unwrap()
}

async fn assert_generic_422(r: reqwest::Response) {
    assert_eq!(r.status(), 422);
    let body = r.text().await.unwrap();
    assert!(body.contains("não foi possível processar o documento"), "{body}");
    assert!(!body.contains("/tmp") && !body.contains("sinal") && !body.contains("worker"), "vazou detalhe interno: {body}");
}

#[tokio::test]
async fn hung_worker_is_killed_at_the_deadline_and_slots_are_freed() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("pid");
    let script = fake_worker(dir.path(), &format!("echo $$ > {}\nexec sleep 300", pidfile.display()));
    let app = TestApp::start_with(|c| {
        c.ocr_worker_exe = script;
        c.ocr_timeout = Duration::from_secs(1);
        c.ocr_max_concurrent = 1;
    })
    .await;
    let (token, _) = app.tokens().await;

    let started = Instant::now();
    let r = upload(&app, &token).await;
    assert_eq!(r.status(), 503);
    assert!(started.elapsed() < Duration::from_secs(15), "não respeitou o prazo: {:?}", started.elapsed());

    // O processo morreu de fato (antes, a thread de OCR seguia rodando depois do timeout).
    let pid = std::fs::read_to_string(&pidfile).unwrap().trim().to_string();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let state = std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default();
    assert!(state.is_empty() || state.contains(") Z "), "worker {pid} ainda vivo: {state}");

    // A única vaga global foi devolvida.
    assert_eq!(app.state.ocr.slots.available_permits(), 1);
}

#[tokio::test]
async fn crashed_worker_gives_generic_error_alert_and_the_api_stays_up() {
    let dir = tempfile::tempdir().unwrap();
    let script = fake_worker(dir.path(), "kill -SEGV $$");
    let app = TestApp::start_with(|c| c.ocr_worker_exe = script).await;
    let (token, _) = app.tokens().await;

    assert_generic_422(upload(&app, &token).await).await;
    assert_eq!(events(&app, "ocr_worker_crashed").await, 1);
    assert_eq!(events(&app, "security_alert").await, 1, "morte por sinal deve gerar alerta");
    assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
}

#[tokio::test]
async fn garbage_or_oversized_reply_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    // Um worker comprometido tentando injetar texto próprio na resposta da API.
    let script = fake_worker(dir.path(), r#"printf '{"status":"rejected","reason":"<script>alert(1)</script>"}'"#);
    let app = TestApp::start_with(|c| c.ocr_worker_exe = script).await;
    let (token, _) = app.tokens().await;
    assert_generic_422(upload(&app, &token).await).await;
    assert_eq!(events(&app, "ocr_worker_bad_reply").await, 1);
}

#[tokio::test]
async fn worker_gets_an_empty_environment_without_secrets() {
    let dir = tempfile::tempdir().unwrap();
    let envfile = dir.path().join("env.txt");
    let script = fake_worker(
        dir.path(),
        &format!(r#"env > {}
printf '{{"status":"ok","documents":[]}}'"#, envfile.display()),
    );
    let app = TestApp::start_with(|c| c.ocr_worker_exe = script).await;
    let (token, _) = app.tokens().await;
    assert_eq!(upload(&app, &token).await.status(), 200);

    let env = std::fs::read_to_string(&envfile).unwrap();
    let keys: Vec<&str> = env.lines().filter_map(|l| l.split_once('=').map(|(k, _)| k)).collect();
    // O processo de teste tem CARGO_*, TEST_ADMIN_DATABASE_URL (com senha!) etc.: nada disso passa.
    assert!(std::env::var("TEST_ADMIN_DATABASE_URL").is_ok());
    for k in &keys {
        assert!(
            !k.starts_with("CARGO") && !k.contains("DATABASE") && !k.contains("SECRET") && !k.contains("PEPPER") && !k.contains("REDIS"),
            "variável vazou para o worker: {k}"
        );
    }
    assert!(!env.contains(&admin_url()), "a URL do banco (com senha) vazou para o worker");
}

#[tokio::test]
async fn real_worker_out_of_memory_is_contained() {
    // Teto de 64 MiB: o motor não cabe, o worker morre; a API responde 422 genérico e segue de pé.
    let app = TestApp::start_with(|c| c.ocr_worker_memory_mb = 64).await;
    let (token, _) = app.tokens().await;
    assert_generic_422(upload(&app, &token).await).await;
    assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
    let failed: i64 = sqlx::query_scalar("SELECT count(*) FROM audit_log WHERE kind LIKE 'ocr_worker_%'").fetch_one(&app.db).await.unwrap();
    assert_eq!(failed, 1);
}

#[tokio::test]
async fn worker_over_its_memory_budget_is_killed_by_the_watchdog() {
    // Worker que aloca ~60 MB e fica parado; teto de 32 MiB de RSS. O vigia o mata em ~100 ms,
    // sem negar nenhuma alocação (o que faria o ONNX Runtime travar): o processo leva SIGKILL.
    let dir = tempfile::tempdir().unwrap();
    let script = fake_worker(dir.path(), "x=$(head -c 60000000 /dev/zero | tr '\\0' a)\nsleep 60");
    let app = TestApp::start_with(|c| {
        c.ocr_worker_exe = script;
        c.ocr_worker_max_rss_mb = 32;
    })
    .await;
    let (token, _) = app.tokens().await;
    let started = Instant::now();
    assert_generic_422(upload(&app, &token).await).await;
    assert!(started.elapsed() < Duration::from_secs(10), "o vigia não agiu: {:?}", started.elapsed());
    assert_eq!(events(&app, "ocr_worker_memory_killed").await, 1);
    assert_eq!(events(&app, "ocr_worker_crashed").await, 0, "morte pelo vigia não é crash");
    assert_eq!(app.state.ocr.slots.available_permits(), 2, "vagas devolvidas");
    assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
}
