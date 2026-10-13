//! `GET /v1/chunk/strategies` e `POST /v1/chunk`, de ponta a ponta: autenticação, validação,
//! limites, todas as estratégias disponíveis passando pelo worker real e o worker isolado
//! contendo entradas hostis.

mod common;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};

/// Tokenizador pequeno de teste (all-MiniLM-L6-v2: `tokenizer.json` + `sentence_bert_config.json`).
fn minilm() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../src/chunking/tests/fixtures/minilm"))
}

const TEXT: &str = "# Manual\n\nPrimeiro parágrafo do manual, com algumas frases. Ele explica o básico. \
E segue um pouco mais para ocupar espaço.\n\n## Instalação\n\nSegundo parágrafo: baixe o pacote, \
descompacte e rode o instalador. Depois reinicie.\n\n| Coluna | Valor |\n|---|---|\n| a | 1 |\n| b | 2 |\n\n\
Qual é o prazo?\nO prazo é de trinta dias corridos.\n\nArt. 1º Esta lei entra em vigor na data da publicação.\n\
Art. 2º Revogam-se as disposições em contrário.";

async fn post(app: &TestApp, token: &str, body: &Value) -> reqwest::Response {
    app.http.post(app.url("/v1/chunk")).bearer_auth(token).json(body).send().await.unwrap()
}

async fn ok(r: reqwest::Response) -> Value {
    let status = r.status();
    let body = r.text().await.unwrap();
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).unwrap()
}

/// Status esperado + corpo no formato de erro da API, sem detalhes internos.
async fn fails(r: reqwest::Response, status: u16) -> String {
    assert_eq!(r.status(), status);
    let v: Value = r.json().await.unwrap();
    let msg = v["error_description"].as_str().unwrap().to_string();
    assert!(!msg.contains("/tmp") && !msg.contains("src/") && !msg.contains("worker") && !msg.contains("panicked"), "{msg}");
    msg
}

async fn strategies(app: &TestApp, token: &str) -> Value {
    ok(app.http.get(app.url("/v1/chunk/strategies")).bearer_auth(token).send().await.unwrap()).await
}

fn find<'a>(catalog: &'a Value, name: &str) -> &'a Value {
    catalog["strategies"].as_array().unwrap().iter().find(|s| s["name"] == name).unwrap_or_else(|| panic!("{name}"))
}

#[tokio::test]
async fn lists_strategies_with_parameters_and_availability() {
    let app = TestApp::start().await;
    assert_eq!(app.http.get(app.url("/v1/chunk/strategies")).send().await.unwrap().status(), 401);
    let (token, _) = app.tokens().await;
    let c = strategies(&app, &token).await;

    assert_eq!(c["count"], 33);
    let rec = find(&c, "recursive");
    assert_eq!(rec["available"], true);
    assert_eq!(rec["chunk_size"], json!({"unit": "characters", "default": 1000, "min": 1, "max": 100000}));
    assert_eq!(rec["chunk_overlap"], json!({"default": 100, "min": 0, "max_fraction_of_chunk_size": 0.5}));
    assert!(rec["aliases"].as_array().unwrap().contains(&json!("recursive_character")));
    // Sem tokenizador configurado, as estratégias medidas por ele ficam indisponíveis (e dizem por quê).
    for name in ["token", "semchunk", "token_sentence_transformer", "docling_hybrid"] {
        let s = find(&c, name);
        assert_eq!(s["available"], false, "{name}");
        assert!(s["unavailable_reason"].as_str().unwrap().contains("CHUNK_TOKENIZER"), "{name}");
    }
    for name in ["semantic", "contextual_retrieval", "late_chunking"] {
        assert_eq!(find(&c, name)["available"], false, "{name}");
    }
    assert_eq!(find(&c, "token_tiktoken")["chunk_size"]["unit"], "tokens");
    assert!(find(&c, "docling_hybrid")["chunk_overlap"].is_null(), "docling_hybrid não usa sobreposição");
    assert_eq!(c["limits"]["max_chunks"], 20000);
}

#[tokio::test]
async fn chunks_text_with_the_chosen_strategy_size_and_overlap() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;

    // Apelido aceito; resposta com o nome canônico e os parâmetros efetivos.
    let v = ok(post(&app, &token, &json!({"strategy": "recursive_character", "chunk_size": 80, "chunk_overlap": 20, "text": TEXT})).await).await;
    assert_eq!((v["strategy"].as_str(), v["unit"].as_str(), v["chunk_size"].as_u64(), v["chunk_overlap"].as_u64()), (Some("recursive"), Some("characters"), Some(80), Some(20)));
    let chunks = v["chunks"].as_array().unwrap();
    assert_eq!(v["count"].as_u64().unwrap() as usize, chunks.len());
    assert!(chunks.len() > 5);
    for c in chunks {
        assert!(c["page_content"].as_str().unwrap().chars().count() <= 80, "{c}");
    }

    // Padrões da estratégia quando nada é informado: o texto inteiro cabe num chunk de 1000.
    let v = ok(post(&app, &token, &json!({"strategy": "recursive", "text": TEXT})).await).await;
    assert_eq!((v["chunk_size"].as_u64(), v["chunk_overlap"].as_u64(), v["count"].as_u64()), (Some(1000), Some(100), Some(1)));

    // Metadados da estratégia.
    let v = ok(post(&app, &token, &json!({"strategy": "markdown_header", "text": TEXT})).await).await;
    assert!(v["chunks"].as_array().unwrap().iter().any(|c| c["metadata"]["h2"] == "Instalação"), "{v}");
    let v = ok(post(&app, &token, &json!({"strategy": "parent_child", "chunk_size": 100, "text": TEXT})).await).await;
    let child = &v["chunks"][0]["metadata"];
    assert!(child["parent_id"].is_string() && child["parent_content"].is_string(), "o filho leva o pai: {child}");

    // Medida em tokens (tiktoken embutido, sem rede).
    let v = ok(post(&app, &token, &json!({"strategy": "tiktoken", "chunk_size": 16, "chunk_overlap": 4, "text": TEXT})).await).await;
    assert_eq!((v["strategy"].as_str(), v["unit"].as_str()), (Some("token_tiktoken"), Some("tokens")));
    assert!(v["count"].as_u64().unwrap() > 5);
    // Texto do cliente é texto: um "token especial" não derruba a requisição.
    ok(post(&app, &token, &json!({"strategy": "token_tiktoken", "text": "fim <|endoftext|> aqui"})).await).await;
}

#[tokio::test]
async fn every_available_strategy_runs_in_the_worker() {
    let app = TestApp::start_with(|c| c.chunk.tokenizer = Some(minilm())).await;
    let (token, _) = app.tokens().await;
    let c = strategies(&app, &token).await;
    let available: Vec<&Value> = c["strategies"].as_array().unwrap().iter().filter(|s| s["available"] == true).collect();
    // As 33, menos as que dependem de embeddings/LLM.
    assert_eq!(available.len(), 30);
    let st = find(&c, "token_sentence_transformer");
    assert_eq!((st["chunk_size"]["default"].as_u64(), st["chunk_size"]["max"].as_u64()), (Some(256), Some(256)), "limite do modelo");

    for s in available {
        let name = s["name"].as_str().unwrap();
        let size = if s["chunk_size"]["unit"] == "tokens" { 24 } else { 120 };
        let mut body = json!({"strategy": name, "chunk_size": size, "text": TEXT});
        if !s["chunk_overlap"].is_null() {
            body["chunk_overlap"] = json!(size / 4);
        }
        let v = ok(post(&app, &token, &body).await).await;
        assert_eq!(v["strategy"], *name);
        assert!(v["count"].as_u64().unwrap() >= 1, "{name}: {v}");
        assert!(v["chunks"].as_array().unwrap().iter().all(|c| !c["page_content"].as_str().unwrap().trim().is_empty()), "{name}");
    }

    // Sobreposição só onde se aplica.
    let m = fails(post(&app, &token, &json!({"strategy": "docling_hybrid", "chunk_overlap": 10, "text": TEXT})).await, 422).await;
    assert!(m.contains("não se aplica"), "{m}");
    let v = ok(post(&app, &token, &json!({"strategy": "docling_hybrid", "chunk_overlap": 0, "text": TEXT})).await).await;
    assert!(v["chunk_overlap"].is_null());
}

#[tokio::test]
async fn data_preparation_output_goes_straight_into_chunking() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let texto = "O contrato vale por doze meses. O pagamento é mensal. A rescisão exige aviso prévio de trinta dias.";
    let form = Form::new().part("file", Part::bytes(minimal_docx(texto)).file_name("contrato.docx"));
    let extracted = ok(app.http.post(app.url("/v1/data_preparation")).bearer_auth(&token).multipart(form).send().await.unwrap()).await;

    let v = ok(post(&app, &token, &json!({"strategy": "sentence", "chunk_size": 40, "chunk_overlap": 0, "documents": extracted["documents"]})).await).await;
    let chunks = v["chunks"].as_array().unwrap();
    assert!(chunks.len() >= 3, "{v}");
    for c in chunks {
        assert_eq!(c["metadata"]["source"], "contrato.docx", "metadados do documento seguem para o chunk");
    }
    let joined: String = chunks.iter().map(|c| c["page_content"].as_str().unwrap()).collect::<Vec<_>>().join(" ");
    assert!(joined.contains("doze meses") && joined.contains("aviso prévio"));
}

#[tokio::test]
async fn invalid_requests_are_rejected_without_details() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;

    // Escopo: um token só com `profile` não divide texto nem lista estratégias.
    api_ocr::store::create_client(&app.db, "so-profile", "SoProfile", &[REDIRECT.to_string()], &["profile".to_string()]).await.unwrap();
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    let profile = app.state.tokens.issue(uid, "so-profile", "profile").unwrap();
    fails(post(&app, &profile, &json!({"strategy": "recursive", "text": "x"})).await, 403).await;
    fails(app.http.get(app.url("/v1/chunk/strategies")).bearer_auth(&profile).send().await.unwrap(), 403).await;
    fails(app.http.post(app.url("/v1/chunk")).json(&json!({"strategy": "recursive", "text": "x"})).send().await.unwrap(), 401).await;

    for (body, status, expected) in [
        (json!({"strategy": "nao-existe", "text": TEXT}), 422, "desconhecida"),
        (json!({"strategy": "token", "text": TEXT}), 422, "indisponível"),
        (json!({"strategy": "semantic", "text": TEXT}), 422, "indisponível"),
        (json!({"strategy": "recursive", "chunk_size": 0, "text": TEXT}), 422, "chunk_size"),
        (json!({"strategy": "recursive", "chunk_size": 100001, "text": TEXT}), 422, "chunk_size"),
        (json!({"strategy": "recursive", "chunk_size": 100, "chunk_overlap": 51, "text": TEXT}), 422, "metade"),
        (json!({"strategy": "recursive", "text": "  \n "}), 422, "vazio"),
        (json!({"strategy": "recursive", "documents": []}), 422, "vazia"),
        (json!({"strategy": "recursive"}), 400, "um dos dois"),
        (json!({"strategy": "recursive", "text": "a", "documents": [{"page_content": "b"}]}), 400, "um dos dois"),
        (json!({"strategy": "recursive", "chunk_size": -5, "text": TEXT}), 400, "JSON"),
        // Opções internas (modelo, URL de LLM, caminhos) não são aceitas do cliente.
        (json!({"strategy": "recursive", "text": TEXT, "options": {"llm_base_url": "http://169.254.169.254"}}), 400, "JSON"),
        (json!({"strategy": "recursive", "text": TEXT, "model_name": "/etc/passwd"}), 400, "JSON"),
        (json!({"strategy": "recursive", "documents": [{"page_content": "a", "extra": 1}]}), 400, "JSON"),
    ] {
        let m = fails(post(&app, &token, &body).await, status).await;
        assert!(m.contains(expected), "{body} => {m}");
    }
    let r = app.http.post(app.url("/v1/chunk")).bearer_auth(&token).header("content-type", "application/json").body("{nao json").send().await.unwrap();
    fails(r, 400).await;
    let r = app.http.post(app.url("/v1/chunk")).bearer_auth(&token).header("content-type", "text/plain").body(r#"{"strategy":"recursive","text":"x"}"#).send().await.unwrap();
    assert!(fails(r, 400).await.contains("Content-Type"));
}

#[tokio::test]
async fn limits_on_text_size_documents_and_chunks() {
    let app = TestApp::start_with(|c| {
        c.chunk.max_text_bytes = 2000;
        c.chunk.max_chunks = 10;
        c.chunk.max_documents = 3;
    })
    .await;
    let (token, _) = app.tokens().await;

    fails(post(&app, &token, &json!({"strategy": "recursive", "text": "a".repeat(2001)})).await, 413).await;
    // Corpo muito acima do teto: cortado na leitura, sem bufferizar tudo.
    fails(post(&app, &token, &json!({"strategy": "recursive", "text": "a".repeat(3 * 1024 * 1024)})).await, 413).await;
    let docs: Vec<Value> = (0..4).map(|i| json!({"page_content": format!("doc {i}")})).collect();
    fails(post(&app, &token, &json!({"strategy": "recursive", "documents": docs})).await, 422).await;
    let big = json!({"page_content": "abc", "metadata": {"x": "y".repeat(5000)}});
    assert!(fails(post(&app, &token, &json!({"strategy": "recursive", "documents": [big]})).await, 422).await.contains("metadata"));

    // Estimativa na API (sem gastar worker): 1900 caracteres em chunks de 10 => ~190 chunks.
    let m = fails(post(&app, &token, &json!({"strategy": "recursive", "chunk_size": 10, "chunk_overlap": 0, "text": "ab ".repeat(633)})).await, 422).await;
    assert!(m.contains("chunks demais"), "{m}");
    // Contagem real no worker: ~20 chunks passam pela estimativa (margem de 2x) e são recusados lá.
    let m = fails(post(&app, &token, &json!({"strategy": "recursive", "chunk_size": 20, "chunk_overlap": 0, "text": "palavra ".repeat(50)})).await, 422).await;
    assert!(m.contains("chunks demais"), "{m}");
    // Em tokens não há estimativa barata: só a do worker.
    let m = fails(post(&app, &token, &json!({"strategy": "token_tiktoken", "chunk_size": 2, "chunk_overlap": 0, "text": "palavra ".repeat(100)})).await, 422).await;
    assert!(m.contains("chunks demais"), "{m}");
}

#[tokio::test]
async fn one_chunking_per_user_at_a_time() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();
    let guard = app.state.chunk.try_user_guard(uid).expect("livre");
    let r = post(&app, &token, &json!({"strategy": "recursive", "text": TEXT})).await;
    assert_eq!(r.status(), 429);
    assert!(r.headers().contains_key("retry-after"));
    // Um OCR em andamento não bloqueia o chunking (vagas separadas), e vice-versa.
    drop(guard);
    let _ocr = app.state.try_user_guard(uid).expect("livre");
    ok(post(&app, &token, &json!({"strategy": "recursive", "text": TEXT})).await).await;
}

#[tokio::test]
async fn hostile_text_only_hurts_the_worker() {
    let app = TestApp::start_with(|c| c.chunk.timeout = Duration::from_secs(5)).await;
    let (token, _) = app.tokens().await;
    // Aninhamento profundo: estoura a pilha dos parsers recursivos (JSON, conversão do DOM) e deixa
    // o parser HTML quadrático (200 mil `<div>` levam ~2 min). O worker morre ou é morto no
    // prazo; a API responde e segue de pé.
    for (strategy, text) in [
        ("json", format!("{}{}", "[".repeat(400_000), "]".repeat(400_000))),
        ("html_semantic", "<div>".repeat(200_000)),
        ("html_header", "<b>".repeat(200_000)),
    ] {
        let started = Instant::now();
        let r = post(&app, &token, &json!({"strategy": strategy, "chunk_size": 100000, "text": text})).await;
        assert!(matches!(r.status().as_u16(), 200 | 422 | 503), "{strategy}: {}", r.status());
        assert!(started.elapsed() < Duration::from_secs(15), "{strategy}: não respeitou o prazo ({:?})", started.elapsed());
        // A API continua de pé e as vagas foram devolvidas.
        assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
        assert_eq!(app.state.chunk.slots.available_permits(), app.state.cfg.chunk.max_concurrent);
    }
}

#[tokio::test]
async fn crashed_chunk_worker_gives_a_generic_error_and_an_alert() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("worker.sh");
    std::fs::write(&script, "#!/bin/sh\nkill -SEGV $$\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let app = TestApp::start_with(|c| c.ocr_worker_exe = script).await;
    let (token, _) = app.tokens().await;

    let m = fails(post(&app, &token, &json!({"strategy": "recursive", "text": TEXT})).await, 422).await;
    assert!(m.contains("não foi possível dividir"), "{m}");
    let n: i64 = sqlx::query_scalar("SELECT count(*) FROM audit_log WHERE kind = 'chunk_worker_crashed'").fetch_one(&app.db).await.unwrap();
    assert_eq!(n, 1);
    assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
}
