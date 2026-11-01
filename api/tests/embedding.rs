//! `GET /v1/embedding/strategies` e `POST /v1/embedding`, de ponta a ponta, sem baixar nada:
//! estratégias sem modelo pelo worker real, validação, limites e o worker isolado. O teste com
//! modelo usa o all-MiniLM-L6-v2 do cache local do Hugging Face, se existir (senão, é pulado).

mod common;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};

const DOCS: [&str; 4] = [
    "Fortaleza possui praias urbanas e dunas no litoral do Ceará.",
    "Rust é uma linguagem de programação rápida e segura.",
    "A receita de bolo leva farinha, ovos e açúcar.",
    "As praias de Fortaleza recebem turistas o ano todo.",
];

async fn post(app: &TestApp, token: &str, body: &Value) -> reqwest::Response {
    app.http.post(app.url("/v1/embedding")).bearer_auth(token).json(body).send().await.unwrap()
}

async fn ok(r: reqwest::Response) -> Value {
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).unwrap()
}

async fn fails(r: reqwest::Response, status: u16) -> String {
    assert_eq!(r.status(), status);
    let v: Value = r.json().await.unwrap();
    let msg = v["error_description"].as_str().unwrap().to_string();
    assert!(!msg.contains("/tmp") && !msg.contains("src/") && !msg.contains("worker") && !msg.contains("panicked"), "{msg}");
    msg
}

fn dense(v: &Value) -> Vec<Vec<f32>> {
    serde_json::from_value(v.clone()).unwrap()
}

/// Produto escalar entre vetores esparsos `{indices, values}`.
fn sparse_dot(a: &Value, b: &Value) -> f32 {
    let pairs = |v: &Value| -> Vec<(u64, f32)> {
        v["indices"].as_array().unwrap().iter().zip(v["values"].as_array().unwrap()).map(|(i, x)| (i.as_u64().unwrap(), x.as_f64().unwrap() as f32)).collect()
    };
    let b = pairs(b);
    pairs(a).iter().filter_map(|(i, x)| b.iter().find(|(j, _)| j == i).map(|(_, y)| x * y)).sum()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (n(a) * n(b))
}

#[tokio::test]
async fn lists_strategies_and_availability() {
    let app = TestApp::start().await;
    assert_eq!(app.http.get(app.url("/v1/embedding/strategies")).send().await.unwrap().status(), 401);
    let (token, _) = app.tokens().await;
    let c = ok(app.http.get(app.url("/v1/embedding/strategies")).bearer_auth(&token).send().await.unwrap()).await;
    assert_eq!(c["count"], 19);
    let find = |n: &str| c["strategies"].as_array().unwrap().iter().find(|s| s["name"] == n).unwrap().clone();
    for n in ["hashing", "tfidf", "bm25", "lsa"] {
        assert_eq!(find(n)["available"], true, "{n}");
    }
    assert_eq!(find("bm25")["fitted_per_request"], true);
    assert_eq!(find("hashing")["default_format"], "sparse");
    for n in ["e5", "bge", "clip", "sparse"] {
        let s = find(n);
        assert_eq!(s["available"], false, "{n}");
        assert!(s["unavailable_reason"].as_str().unwrap().contains("EMBEDDING_MODELS"), "{n}");
    }
    assert!(find("ollama")["unavailable_reason"].as_str().unwrap().contains("rede"));
    assert_eq!(find("clip")["accepts_images"], true);
}

#[tokio::test]
async fn logic_strategies_run_in_the_worker() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;

    // hashing: esparso por padrão; denso, com corte e norma 1, se pedido.
    let v = ok(post(&app, &token, &json!({"strategy": "hashing", "texts": DOCS})).await).await;
    assert_eq!((v["format"].as_str(), v["dimension"].as_u64(), v["count"].as_u64()), (Some("sparse"), Some(4096), Some(4)));
    assert!(v["embeddings"].is_null() && v["sparse_embeddings"].as_array().unwrap().len() == 4);
    let v = ok(post(&app, &token, &json!({"strategy": "hashing", "texts": DOCS, "format": "dense", "dimensions": 128, "normalize": true})).await).await;
    let e = dense(&v["embeddings"]);
    assert_eq!((e.len(), e[0].len()), (4, 128));
    assert!(e.iter().all(|r| (r.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 1e-5));

    // bm25: acervo e consultas na mesma requisição; a consulta acha os documentos certos.
    let v = ok(post(&app, &token, &json!({"strategy": "bm25", "texts": DOCS, "queries": ["praias Fortaleza", "bolo"]})).await).await;
    let docs = v["sparse_embeddings"].as_array().unwrap();
    let qs = v["sparse_query_embeddings"].as_array().unwrap();
    let scores: Vec<f32> = docs.iter().map(|d| sparse_dot(&qs[0], d)).collect();
    assert!(scores[0] > 0.0 && scores[3] > 0.0 && scores[1] == 0.0 && scores[2] == 0.0, "{scores:?}");
    assert!(sparse_dot(&qs[1], &docs[2]) > 0.0);

    // tfidf e lsa: consultas no mesmo espaço dos documentos.
    let v = ok(post(&app, &token, &json!({"strategy": "tfidf", "texts": DOCS, "queries": ["linguagem rápida"], "format": "dense"})).await).await;
    let (d, q) = (dense(&v["embeddings"]), dense(&v["query_embeddings"]));
    assert_eq!(d[0].len(), q[0].len());
    let best = (0..4).max_by(|&a, &b| cos(&q[0], &d[a]).partial_cmp(&cos(&q[0], &d[b])).unwrap()).unwrap();
    assert_eq!(best, 1);
    let v = ok(post(&app, &token, &json!({"strategy": "lsa", "texts": DOCS, "queries": ["praias"], "dimensions": 3})).await).await;
    assert_eq!((v["format"].as_str(), v["dimension"].as_u64()), (Some("dense"), Some(3)));
}

#[tokio::test]
async fn chunk_output_goes_straight_into_embedding() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let text = DOCS.join("\n\n");
    let chunks = ok(app.http.post(app.url("/v1/chunk")).bearer_auth(&token).json(&json!({"strategy": "paragraph", "chunk_size": 80, "chunk_overlap": 0, "text": text})).send().await.unwrap()).await;
    let n = chunks["count"].as_u64().unwrap();
    assert!(n >= 4);
    let v = ok(post(&app, &token, &json!({"strategy": "hashing", "documents": chunks["chunks"]})).await).await;
    assert_eq!(v["count"].as_u64(), Some(n));
}

#[tokio::test]
async fn invalid_requests_are_rejected_without_details() {
    let app = TestApp::start_with(|c| {
        c.embedding.max_texts = 3;
        c.embedding.max_text_bytes = 1000;
    })
    .await;
    let (token, _) = app.tokens().await;
    api_ocr::store::create_client(&app.db, "so-profile", "SoProfile", &[REDIRECT.to_string()], &["profile".to_string()]).await.unwrap();
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    let profile = app.state.tokens.issue(uid, "so-profile", "profile").unwrap();
    fails(post(&app, &profile, &json!({"strategy": "hashing", "texts": ["x"]})).await, 403).await;

    for (body, status, expected) in [
        (json!({"strategy": "openai", "texts": ["x"]}), 422, "pago"),
        (json!({"strategy": "nao-existe", "texts": ["x"]}), 422, "desconhecida"),
        (json!({"strategy": "e5", "texts": ["x"]}), 422, "indisponível"),
        (json!({"strategy": "ollama", "texts": ["x"]}), 422, "indisponível"),
        (json!({"strategy": "hashing", "texts": []}), 422, "vazia"),
        (json!({"strategy": "hashing", "texts": ["a", "b", "c", "d"]}), 422, "demais"),
        (json!({"strategy": "hashing", "texts": ["x".repeat(1001)]}), 413, "maior"),
        (json!({"strategy": "hashing", "texts": ["x"], "images": ["aGk="]}), 400, "um deles"),
        (json!({"strategy": "hashing", "images": ["aGk="]}), 422, "imagens"),
        (json!({"strategy": "hashing", "texts": ["x"], "queries": ["y"]}), 422, "queries"),
        (json!({"strategy": "bm25", "texts": ["x"], "input_type": "query"}), 422, "acervo"),
        (json!({"strategy": "hashing", "texts": ["x"], "input_type": "pergunta"}), 422, "input_type"),
        (json!({"strategy": "hashing", "texts": ["x"], "dimensions": 0}), 422, "dimensions"),
        (json!({"strategy": "hashing", "texts": ["x"], "format": "csv"}), 422, "format"),
        // Opções internas (caminho de modelo, host do Ollama) não são aceitas do cliente.
        (json!({"strategy": "hashing", "texts": ["x"], "model_name": "/etc"}), 400, "JSON"),
        (json!({"strategy": "hashing", "texts": ["x"], "options": {"host": "http://169.254.169.254"}}), 400, "JSON"),
    ] {
        let m = fails(post(&app, &token, &body).await, status).await;
        assert!(m.contains(expected), "{body} => {m}");
    }
}

#[tokio::test]
async fn one_embedding_per_user_at_a_time() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    let guard = app.state.embedding.try_user_guard(uid).expect("livre");
    assert_eq!(post(&app, &token, &json!({"strategy": "hashing", "texts": ["x"]})).await.status(), 429);
    drop(guard);
    ok(post(&app, &token, &json!({"strategy": "hashing", "texts": ["x"]})).await).await;
}

#[tokio::test]
async fn crashed_embedding_worker_gives_a_generic_error_and_an_alert() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("worker.sh");
    std::fs::write(&script, "#!/bin/sh\nkill -SEGV $$\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let app = TestApp::start_with(|c| c.ocr_worker_exe = script).await;
    let (token, _) = app.tokens().await;
    let m = fails(post(&app, &token, &json!({"strategy": "hashing", "texts": ["x"]})).await, 422).await;
    assert!(m.contains("não foi possível gerar"), "{m}");
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM audit_log WHERE kind = 'embedding_worker_crashed'").fetch_one(&app.db).await.unwrap();
    assert_eq!(n, 1);
}

/// all-MiniLM-L6-v2 já no cache local (com o export ONNX), sem rede.
fn cached_minilm() -> Option<PathBuf> {
    let home = std::env::var_os("HF_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface")))?;
    let snapshots = home.join("hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots");
    std::fs::read_dir(snapshots).ok()?.flatten().map(|e| e.path()).find(|p| p.join("onnx/model.onnx").exists() && p.join("tokenizer.json").exists())
}

#[tokio::test]
async fn local_model_runs_in_the_sandboxed_worker() {
    let Some(model) = cached_minilm() else {
        eprintln!("AVISO: all-MiniLM-L6-v2 (com onnx/model.onnx) não está no cache local; teste pulado");
        return;
    };
    let app = TestApp::start_with(|c| {
        c.embedding.models.insert("sentence_transformer".into(), model.clone());
    })
    .await;
    let (token, _) = app.tokens().await;
    let c = ok(app.http.get(app.url("/v1/embedding/strategies")).bearer_auth(&token).send().await.unwrap()).await;
    let st = c["strategies"].as_array().unwrap().iter().find(|s| s["name"] == "sentence_transformer").unwrap().clone();
    assert_eq!(st["available"], true);
    assert_eq!(st["model"], "sentence-transformers/all-MiniLM-L6-v2", "nome do modelo sem o caminho do servidor");

    let en = ["Fortaleza has urban beaches and dunes.", "Rust is a fast and safe programming language.", "The cake recipe needs flour and eggs."];
    let v = ok(post(&app, &token, &json!({"strategy": "st", "texts": en})).await).await;
    assert_eq!((v["dimension"].as_u64(), v["format"].as_str()), (Some(384), Some("dense")));
    let d = dense(&v["embeddings"]);
    let q = ok(post(&app, &token, &json!({"strategy": "sentence_transformer", "texts": ["safe programming language"], "input_type": "query"})).await).await;
    let q = dense(&q["embeddings"]);
    let best = (0..3).max_by(|&a, &b| cos(&q[0], &d[a]).partial_cmp(&cos(&q[0], &d[b])).unwrap()).unwrap();
    assert_eq!(best, 1);
}
