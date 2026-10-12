#![allow(dead_code)]
use std::net::SocketAddr;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as B64;
use api_ocr::config::Config;
use api_ocr::{crypto, state::AppState, store};
use sha2::{Digest, Sha256};
use sqlx::{Connection, PgConnection, PgPool};
use uuid::Uuid;

pub const ISSUER: &str = "http://localhost";
pub const CLIENT_ID: &str = "test-client";
pub const REDIRECT: &str = "http://localhost:3000/callback";
pub const EMAIL: &str = "ana@exemplo.com";
pub const PASSWORD: &str = "uma-senha-bem-longa-42";

pub fn admin_url() -> String {
    std::env::var("TEST_ADMIN_DATABASE_URL").expect("defina TEST_ADMIN_DATABASE_URL (ver api/README.md)")
}

pub fn base_config(db_url: String) -> Config {
    Config {
        database_url: db_url,
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        issuer: ISSUER.to_string(),
        jwt_secret: b"test-jwt-secret-test-jwt-secret-0123456789".to_vec(),
        jwt_previous_secrets: vec![],
        password_pepper: b"test-pepper-test-pepper-test-pepper-98765".to_vec(),
        trust_proxy: false,
        allow_registration: true,
        enable_docs: false,
        cors_origins: vec![],
        access_token_ttl: Duration::from_secs(600),
        refresh_token_ttl: Duration::from_secs(86_400),
        refresh_family_max: Duration::from_secs(7 * 86_400),
        auth_code_ttl: Duration::from_secs(60),
        ocr_max_upload_bytes: 2 * 1024 * 1024,
        ocr_max_pages: 5,
        ocr_max_concurrent: 2,
        ocr_timeout: Duration::from_secs(60),
        ocr_memory_watermark_pct: 100,
        ocr_worker_exe: env!("CARGO_BIN_EXE_api_ocr").into(),
        ocr_worker_memory_mb: 0,
        ocr_worker_max_rss_mb: 0,
        max_connections: 1024,
        max_connections_per_ip: 64,
        max_in_flight: 256,
        rate_general_per_min: 6000,
        rate_auth_per_min: 6000,
        rate_ocr_per_min: 600,
        rate_limit_redis_url: None,
        rate_limit_redis_prefix: "test:".into(),
        security_alert_login_failures: 20,
    }
}

pub struct TestApp {
    pub addr: SocketAddr,
    pub db: PgPool,
    pub state: AppState,
    pub http: reqwest::Client,
    db_name: String,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl TestApp {
    pub async fn start() -> Self {
        Self::start_with(|_| {}).await
    }

    pub async fn start_with(tweak: impl FnOnce(&mut Config)) -> Self {
        let admin = admin_url();
        let db_name = format!("t_{}", Uuid::new_v4().simple());
        let mut conn = PgConnection::connect(&admin).await.expect("admin db");
        sqlx::query(sqlx::AssertSqlSafe(format!("CREATE DATABASE {db_name}"))).execute(&mut conn).await.unwrap();
        let url = {
            let mut u = url::Url::parse(&admin).unwrap();
            u.set_path(&db_name);
            u.to_string()
        };

        // Pool pequeno: dezenas de testes em paralelo não podem esgotar o max_connections do Postgres.
        let db = api_ocr::connect_db_with(&url, 3).await.unwrap();
        api_ocr::MIGRATOR.run(&db).await.unwrap();

        let mut cfg = base_config(url);
        tweak(&mut cfg);

        store::create_client(&db, CLIENT_ID, "Cliente de Teste", &[REDIRECT.to_string()], &["ocr".to_string(), "profile".to_string()])
            .await
            .unwrap();
        let svc = crypto::PasswordService::new(cfg.password_pepper.clone(), 2).unwrap();
        let hash = svc.hash(PASSWORD.to_string()).await.unwrap();
        store::create_user(&db, Uuid::new_v4(), EMAIL, &hash).await.unwrap();

        if cfg.enable_docs {
            api_ocr::docs::ensure_client(&db, &cfg.issuer).await.unwrap();
        }
        let state = api_ocr::build_state(cfg.clone(), db.clone()).await.unwrap();
        let app = api_ocr::router(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            api_ocr::server::serve(listener, app, &cfg, async {
                let _ = rx.await;
            })
            .await;
        });

        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();
        Self { addr, db, state, http, db_name, shutdown: Some(tx) }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.addr, path)
    }

    /// Fluxo de login: devolve o `code` do redirect.
    pub async fn login_for_code(&self, challenge: &str, state: &str) -> String {
        let (csrf, _) = self.authorize_page(challenge, state).await;
        let resp = self.post_login(&csrf, challenge, state, EMAIL, PASSWORD).await;
        assert_eq!(resp.status(), 303, "login deveria redirecionar");
        let loc = resp.headers()["location"].to_str().unwrap().to_string();
        let u = url::Url::parse(&loc).unwrap();
        let q: std::collections::HashMap<_, _> = u.query_pairs().into_owned().collect();
        assert_eq!(q.get("state").map(String::as_str), Some(state));
        assert_eq!(q.get("iss").map(String::as_str), Some(ISSUER));
        q["code"].clone()
    }

    pub fn authorize_query(&self, challenge: &str, state: &str) -> Vec<(&'static str, String)> {
        vec![
            ("response_type", "code".into()),
            ("client_id", CLIENT_ID.into()),
            ("redirect_uri", REDIRECT.into()),
            ("scope", "ocr profile".into()),
            ("state", state.into()),
            ("code_challenge", challenge.into()),
            ("code_challenge_method", "S256".into()),
        ]
    }

    /// GET /oauth/authorize => (csrf, html)
    pub async fn authorize_page(&self, challenge: &str, state: &str) -> (String, String) {
        let resp = self.http.get(self.url("/oauth/authorize")).query(&self.authorize_query(challenge, state)).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        let cookie = resp.headers()["set-cookie"].to_str().unwrap().to_string();
        let csrf = cookie.split(';').next().unwrap().split_once('=').unwrap().1.to_string();
        (csrf, resp.text().await.unwrap())
    }

    pub async fn post_login(&self, csrf: &str, challenge: &str, state: &str, email: &str, password: &str) -> reqwest::Response {
        let mut form = self.authorize_query(challenge, state);
        form.push(("email", email.into()));
        form.push(("password", password.into()));
        form.push(("csrf", csrf.into()));
        self.http
            .post(self.url("/oauth/authorize"))
            .header("cookie", format!("csrf={csrf}"))
            .form(&form)
            .send()
            .await
            .unwrap()
    }

    pub async fn exchange(&self, code: &str, verifier: &str) -> reqwest::Response {
        self.http
            .post(self.url("/oauth/token"))
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", REDIRECT),
                ("client_id", CLIENT_ID),
                ("code_verifier", verifier),
            ])
            .send()
            .await
            .unwrap()
    }

    pub async fn refresh(&self, refresh_token: &str) -> reqwest::Response {
        self.http
            .post(self.url("/oauth/token"))
            .form(&[("grant_type", "refresh_token"), ("refresh_token", refresh_token), ("client_id", CLIENT_ID)])
            .send()
            .await
            .unwrap()
    }

    /// Login + troca do código: devolve (access, refresh).
    pub async fn tokens(&self) -> (String, String) {
        let (verifier, challenge) = pkce();
        let code = self.login_for_code(&challenge, "estado-123").await;
        let resp = self.exchange(&code, &verifier).await;
        assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
        let j: serde_json::Value = resp.json().await.unwrap();
        (j["access_token"].as_str().unwrap().into(), j["refresh_token"].as_str().unwrap().into())
    }
}

impl Drop for TestApp {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let (admin, name) = (admin_url(), self.db_name.clone());
        // Limpeza em thread própria (Drop é síncrono).
        let _ = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
            rt.block_on(async {
                if let Ok(mut c) = PgConnection::connect(&admin).await {
                    let _ = sqlx::query(sqlx::AssertSqlSafe(format!("DROP DATABASE IF EXISTS {name} WITH (FORCE)"))).execute(&mut c).await;
                }
            });
        })
        .join();
    }
}

pub fn pkce() -> (String, String) {
    let verifier = crypto::random_token(48);
    let challenge = B64.encode(Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

/// DOCX mínimo (sem imagens) com o texto dado.
pub fn minimal_docx(text: &str) -> Vec<u8> {
    use std::io::Write;
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut z = zip::ZipWriter::new(&mut buf);
        let o = zip::write::SimpleFileOptions::default();
        let files: [(&str, String); 3] = [
            ("[Content_Types].xml", r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#.into()),
            ("_rels/.rels", r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#.into()),
            ("word/document.xml", format!(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"#)),
        ];
        for (name, body) in files {
            z.start_file(name, o).unwrap();
            z.write_all(body.as_bytes()).unwrap();
        }
        z.finish().unwrap();
    }
    buf.into_inner()
}
