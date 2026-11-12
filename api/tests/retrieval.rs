//! `GET /v1/retrieval/strategies` e `POST /v1/retrieval`, de ponta a ponta pelo worker isolado,
//! sem baixar nada: estratégias lexicais, densas com embedding sem modelo (`hashing`), fusão,
//! consultas geradas pelo cliente, a saída do `/v1/chunk` como acervo, validação e falhas.

mod common;
use std::os::unix::fs::PermissionsExt;

use common::*;
use serde_json::{Value, json};

fn docs() -> Value {
    json!([
        {"page_content": "Fortaleza possui praias urbanas e dunas no litoral do Ceará.", "metadata": {"id": "fortaleza", "kind": "viagem", "timestamp": "2026-11-01T00:00:00Z"}},
        {"page_content": "Rust é uma linguagem de programação rápida e segura.", "metadata": {"id": "rust", "kind": "codigo", "timestamp": "2026-01-01T00:00:00Z"}},
        {"page_content": "A receita de bolo leva farinha, ovos e açúcar.", "metadata": {"id": "bolo", "kind": "receita"}},
        {"page_content": "As praias de Fortaleza recebem turistas o ano todo.", "metadata": {"id": "praias", "kind": "viagem", "timestamp": "2026-10-01T00:00:00Z"}},
        {"page_content": "O artigo 5 da lei 8.078/90 trata do consumidor.", "metadata": {"id": "lei", "kind": "juridico"}},
    ])
}

async fn post(app: &TestApp, token: &str, body: &Value) -> reqwest::Response {
    app.http.post(app.url("/v1/retrieval")).bearer_auth(token).json(body).send().await.unwrap()
}

async fn ok(r: reqwest::Response) -> Value {
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).unwrap()
}

async fn fails(r: reqwest::Response, status: u16) -> String {
    let got = r.status();
    let v: Value = r.json().await.unwrap();
    assert_eq!(got, status, "{v}");
    let msg = v["error_description"].as_str().unwrap().to_string();
    assert!(!msg.contains("/tmp") && !msg.contains("src/") && !msg.contains("worker") && !msg.contains("panicked"), "{msg}");
    msg
}

/// Ids dos documentos do resultado `i`.
fn ids(v: &Value, i: usize) -> Vec<String> {
    v["results"][i]["documents"].as_array().unwrap().iter().map(|d| d["metadata"]["id"].as_str().unwrap_or("").to_string()).collect()
}

#[tokio::test]
async fn lists_strategies_and_availability() {
    let app = TestApp::start().await;
    assert_eq!(app.http.get(app.url("/v1/retrieval/strategies")).send().await.unwrap().status(), 401);
    let (token, _) = app.tokens().await;
    let c = ok(app.http.get(app.url("/v1/retrieval/strategies")).bearer_auth(&token).send().await.unwrap()).await;
    assert_eq!(c["count"], 44);
    let find = |n: &str| c["strategies"].as_array().unwrap().iter().find(|s| s["name"] == n).unwrap().clone();
    for n in ["bm25", "tfidf", "dense", "hybrid", "multi_query", "parent_document", "recency", "fallback", "cascade", "adaptive"] {
        assert_eq!(find(n)["available"], true, "{n}");
    }
    for n in ["rerank", "cross_encoder_rerank", "colbert_rerank", "llm_rerank", "contextual_compression", "graph", "web_search", "router", "custom", "multimodal", "self_query"] {
        let s = find(n);
        assert_eq!(s["available"], false, "{n}");
        assert!(s["unavailable_reason"].is_string(), "{n}");
    }
    assert!(find("rerank")["unavailable_reason"].as_str().unwrap().contains("RETRIEVAL_RERANK_MODEL"));
    assert_eq!(find("hyde")["requires"], json!(["generated_queries", "embedding"]));
    assert_eq!(find("bm25")["requires"], json!([]));
    assert!(find("hybrid")["aliases"].as_array().unwrap().is_empty());
    assert_eq!(find("rrf")["aliases"], json!(["fusion", "reciprocal_rank_fusion"]));
    assert_eq!(c["embedding_strategies"], json!(["tfidf", "hashing", "bm25", "lsa"]));
    assert!(c["rerank_model"].is_null());
    assert_eq!(c["limits"]["max_k"], 100);
}

#[tokio::test]
async fn lexical_dense_and_fused_search_in_the_worker() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;

    // BM25, várias consultas sobre o mesmo acervo.
    let v = ok(post(&app, &token, &json!({"strategy": "bm25", "queries": ["praias de Fortaleza", "bolo"], "documents": docs(), "k": 2})).await).await;
    assert_eq!((v["strategy"].as_str(), v["count"].as_u64(), v["embedding"].is_null()), (Some("bm25"), Some(2), true));
    let first = ids(&v, 0);
    assert_eq!(first.len(), 2);
    assert!(first.contains(&"fortaleza".to_string()) && first.contains(&"praias".to_string()), "{first:?}");
    assert_eq!(ids(&v, 1)[0], "bolo");
    let d = &v["results"][0]["documents"][0];
    assert_eq!((d["metadata"]["retrieval_rank"].as_u64(), d["metadata"]["retrieval_strategy"].as_str()), (Some(1), Some("bm25")));
    assert!(d["metadata"]["retrieval_score"].as_f64().unwrap() > 0.0);
    assert_eq!(v["results"][0]["query"], "praias de Fortaleza");

    // Filtro por metadado, apelido e alteração dos parâmetros do BM25.
    let v = ok(post(&app, &token, &json!({"strategy": "bm25_okapi", "query": "praias", "documents": docs(), "filters": {"kind": ["viagem", "receita"]}, "options": {"k1": 1.2, "b": 0.5}})).await).await;
    assert!(ids(&v, 0).iter().all(|id| ["fortaleza", "praias", "bolo"].contains(&id.as_str())));

    // Densa com embedding sem modelo; híbrida com pesos.
    let emb = json!({"strategy": "hashing", "dimensions": 1024});
    let v = ok(post(&app, &token, &json!({"strategy": "dense", "query": "linguagem de programação segura", "documents": docs(), "embedding": emb, "k": 1})).await).await;
    assert_eq!((ids(&v, 0), v["embedding"].as_str()), (vec!["rust".to_string()], Some("hashing")));
    let v = ok(post(&app, &token, &json!({"strategy": "weighted_hybrid", "query": "lei 8.078/90", "documents": docs(), "embedding": emb, "weights": [0.3, 0.7], "k": 3})).await).await;
    assert_eq!(ids(&v, 0)[0], "lei");
    let v = ok(post(&app, &token, &json!({"strategy": "rrf", "query": "praias", "documents": docs(), "components": ["keyword", "tfidf"], "options": {"rrf_k": 10}})).await).await;
    assert!(ids(&v, 0).contains(&"praias".to_string()));

    // Sem `embedding`, o fallback pula a densa e cai no BM25; o adaptive escolhe o BM25.
    let v = ok(post(&app, &token, &json!({"strategy": "fallback", "query": "bolo", "documents": docs()})).await).await;
    assert_eq!(v["results"][0]["documents"][0]["metadata"]["retrieval_strategy"], "bm25");
    let v = ok(post(&app, &token, &json!({"strategy": "adaptive", "query": "bolo", "documents": docs()})).await).await;
    assert_eq!(v["results"][0]["documents"][0]["metadata"]["retrieval_strategy"], "bm25");

    // Recência sem modelo; cascata com estágios lexicais.
    let v = ok(post(&app, &token, &json!({"strategy": "recency", "query": "qualquer", "documents": docs(), "k": 3})).await).await;
    assert_eq!(ids(&v, 0), ["fortaleza", "praias", "rust"]);
    let v = ok(post(&app, &token, &json!({"strategy": "cascade", "query": "praias", "documents": docs(), "options": {"stages": ["bm25", "keyword"]}})).await).await;
    assert!(!ids(&v, 0).is_empty());
}

#[tokio::test]
async fn generated_queries_and_chunk_output() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;

    // Consultas geradas pelo LLM do cliente, buscadas e fundidas.
    let body = json!({
        "strategy": "multi_query", "query": "doce", "documents": docs(),
        "generated_queries": ["receita de bolo", "farinha e ovos"], "options": {"base_strategy": "bm25"},
    });
    let v = ok(post(&app, &token, &body).await).await;
    assert_eq!(ids(&v, 0)[0], "bolo");
    let body = json!({"strategy": "hyde", "query": "onde tem dunas?", "documents": docs(), "generated_queries": ["Fortaleza tem dunas no litoral"], "options": {"base_strategy": "keyword"}});
    assert_eq!(ids(&ok(post(&app, &token, &body).await).await, 0)[0], "fortaleza");

    // parent_child do chunking → parent_document: busca no trecho, devolve o pai inteiro.
    let text = "Fortaleza tem praias urbanas e jangadas no fim da tarde. ".repeat(4) + &"Rust garante segurança de memória sem coletor de lixo. ".repeat(4);
    let chunk = json!({"strategy": "parent_child", "chunk_size": 60, "chunk_overlap": 0, "text": text});
    let chunks = ok(app.http.post(app.url("/v1/chunk")).bearer_auth(&token).json(&chunk).send().await.unwrap()).await;
    assert!(chunks["count"].as_u64().unwrap() > 2);
    let body = json!({"strategy": "parent_document", "query": "coletor de lixo", "documents": chunks["chunks"], "k": 1, "options": {"base_strategy": "bm25"}});
    let v = ok(post(&app, &token, &body).await).await;
    let parent = v["results"][0]["documents"][0]["page_content"].as_str().unwrap();
    assert!(parent.matches("Rust garante").count() >= 2, "{parent}");
}

#[tokio::test]
async fn invalid_requests_are_rejected_without_details() {
    let app = TestApp::start_with(|c| {
        c.retrieval.max_queries = 2;
        c.retrieval.max_text_bytes = 2000;
        c.retrieval.max_query_bytes = 100;
    })
    .await;
    let (token, _) = app.tokens().await;
    api_ocr::store::create_client(&app.db, "so-profile", "SoProfile", &[REDIRECT.to_string()], &["profile".to_string()]).await.unwrap();
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    let profile = app.state.tokens.issue(uid, "so-profile", "profile").unwrap();
    fails(post(&app, &profile, &json!({"strategy": "bm25", "query": "x", "documents": docs()})).await, 403).await;

    let d = docs();
    let big = json!([{"page_content": "x".repeat(2100)}]);
    for (body, status, expected) in [
        (json!({"strategy": "nao-existe", "query": "x", "documents": d}), 422, "desconhecida"),
        (json!({"strategy": "graph", "query": "x", "documents": d}), 422, "indisponível"),
        (json!({"strategy": "rerank", "query": "x", "documents": d}), 422, "indisponível"),
        (json!({"strategy": "bm25", "documents": d}), 400, "um deles"),
        (json!({"strategy": "bm25", "query": "x", "queries": ["y"], "documents": d}), 400, "um deles"),
        (json!({"strategy": "bm25", "queries": ["a", "b", "c"], "documents": d}), 422, "demais"),
        (json!({"strategy": "bm25", "query": "  ", "documents": d}), 422, "vazia"),
        (json!({"strategy": "bm25", "query": "x".repeat(101), "documents": d}), 413, "maior"),
        (json!({"strategy": "bm25", "query": "x", "documents": []}), 422, "vazia"),
        (json!({"strategy": "bm25", "query": "x", "documents": big}), 413, "maior"),
        (json!({"strategy": "bm25", "query": "x", "documents": d, "k": 0}), 422, "intervalo"),
        (json!({"strategy": "bm25", "query": "x", "documents": d, "fetch_k": 100_000}), 422, "intervalo"),
        (json!({"strategy": "rrf", "query": "x", "documents": d, "weights": [-1]}), 422, "weights"),
        (json!({"strategy": "rrf", "query": "x", "documents": d, "components": ["bm25", "hybrid"]}), 422, "fusão"),
        (json!({"strategy": "rrf", "query": "x", "documents": d, "components": ["bm25", "keyword"], "weights": [1]}), 422, "opções inválidas"),
        (json!({"strategy": "rrf", "query": "x", "documents": d, "components": ["graph"]}), 422, "indisponível"),
        (json!({"strategy": "hyde", "query": "x", "documents": d}), 422, "generated_queries"),
        (json!({"strategy": "bm25", "queries": ["a", "b"], "documents": d, "generated_queries": ["y"]}), 422, "única"),
        (json!({"strategy": "cascade", "query": "x", "documents": d}), 422, "stages"),
        (json!({"strategy": "dense", "query": "x", "documents": d}), 422, "embedding"),
        (json!({"strategy": "dense", "query": "x", "documents": d, "embedding": {"strategy": "e5"}}), 422, "indisponível"),
        (json!({"strategy": "dense", "query": "x", "documents": d, "embedding": {"strategy": "sparse"}}), 422, "indisponível"),
        (json!({"strategy": "dense", "query": "x", "documents": d, "embedding": {"strategy": "hashing", "dimensions": 100_000}}), 422, "dimensions"),
        (json!({"strategy": "bm25", "query": "x", "documents": d, "options": {"k1": 99}}), 422, "options"),
        // Opções e campos do servidor (dispositivo, caminhos de modelo, provider) não vêm do cliente.
        (json!({"strategy": "bm25", "query": "x", "documents": d, "options": {"device": "cuda"}}), 422, "options"),
        (json!({"strategy": "bm25", "query": "x", "documents": d, "rerank_model": "/etc"}), 400, "JSON"),
        (json!({"strategy": "bm25", "query": "x", "documents": d, "provider": "pinecone"}), 400, "JSON"),
        (json!({"strategy": "dense", "query": "x", "documents": d, "embedding": {"strategy": "hashing", "model_name": "/etc"}}), 400, "JSON"),
    ] {
        let m = fails(post(&app, &token, &body).await, status).await;
        assert!(m.contains(expected), "{body} => {m}");
    }
}

#[tokio::test]
async fn one_retrieval_per_user_at_a_time() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    let body = json!({"strategy": "bm25", "query": "praias", "documents": docs()});
    let guard = app.state.retrieval.try_user_guard(uid).expect("livre");
    assert_eq!(post(&app, &token, &body).await.status(), 429);
    drop(guard);
    ok(post(&app, &token, &body).await).await;
}

#[tokio::test]
async fn crashed_retrieval_worker_gives_a_generic_error_and_an_alert() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("worker.sh");
    std::fs::write(&script, "#!/bin/sh\nkill -SEGV $$\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let app = TestApp::start_with(|c| c.ocr_worker_exe = script).await;
    let (token, _) = app.tokens().await;
    let m = fails(post(&app, &token, &json!({"strategy": "bm25", "query": "x", "documents": docs()})).await, 422).await;
    assert!(m.contains("não foi possível buscar"), "{m}");
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM audit_log WHERE kind = 'retrieval_worker_crashed'").fetch_one(&app.db).await.unwrap();
    assert_eq!(n, 1);
}
