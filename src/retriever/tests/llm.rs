//! Cliente compatível com OpenAI contra um servidor HTTP local (sem rede externa).
#![cfg(feature = "llm")]

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;

use retriever::{RetrievalError, RetrievalManager, Retriever};
use serde_json::{Value, json};

/// Atende `replies.len()` pedidos, devolvendo cada resposta e os pedidos recebidos
/// (cabeçalhos em minúsculas e corpo JSON).
fn serve(replies: Vec<(u16, Value)>) -> (String, mpsc::Receiver<(Vec<String>, Value)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1/", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for (status, reply) in replies {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut headers = Vec::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim_end().to_lowercase();
                if line.is_empty() {
                    break;
                }
                if let Some(n) = line.strip_prefix("content-length:") {
                    length = n.trim().parse().unwrap();
                }
                headers.push(line);
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            tx.send((headers, serde_json::from_slice(&body).unwrap())).unwrap();
            let text = reply.to_string();
            let mut out = stream;
            write!(out, "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{text}", text.len()).unwrap();
        }
    });
    (url, rx)
}

fn chat(content: &str) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": content}}]})
}

fn manager(strategy: &str, llm: Value) -> RetrievalManager {
    let docs = json!([
        {"page_content": "Python possui tipagem dinâmica e uma sintaxe simples.", "metadata": {"id": "python"}},
        {"page_content": "Rust é uma linguagem com tipagem estática.", "metadata": {"id": "rust"}},
    ]);
    RetrievalManager::from_value(json!({"strategy": strategy, "documents": docs, "k": 1, "llm": llm}))
        .unwrap()
        .with_embedder(common::fake_embedder())
}

#[test]
fn rephrase_through_openai_compatible_server() {
    let (url, rx) = serve(vec![(200, chat("  rust  "))]);
    let m = manager("rephrase_query", json!({"model": "fake", "base_url": url, "max_tokens": 32, "extra": {"seed": 7}}));
    m.load_retriever().unwrap();
    assert_eq!(m.retrieve("python").unwrap()[0].metadata["id"], "rust");
    let (headers, body) = rx.recv().unwrap();
    assert!(headers[0].starts_with("post /v1/chat/completions "));
    assert!(!headers.iter().any(|h| h.starts_with("authorization")), "sem api_key_env, nenhuma chave");
    assert_eq!(body["model"], "fake");
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["max_tokens"], 32);
    assert_eq!(body["seed"], 7);
    assert_eq!(body["messages"][0]["content"], "Reescreva esta consulta para busca, sem responder:\npython");
}

#[test]
fn api_key_and_errors() {
    let (url, rx) = serve(vec![
        (401, json!({"error": {"message": "chave inválida"}})),
        (200, json!({"choices": [{"message": {"content": "", "reasoning_content": "pensando..."}}]})),
        (200, chat("")),
    ]);
    let m = manager("hyde", json!({"model": "fake", "base_url": url, "api_key": "segredo"}));
    match m.retrieve("python") {
        Err(RetrievalError::Model(msg)) => assert!(msg.contains("401") && msg.contains("chave inválida"), "{msg}"),
        other => panic!("{other:?}"),
    }
    let (headers, _) = rx.recv().unwrap();
    assert!(headers.contains(&"authorization: bearer segredo".to_string()));
    assert!(matches!(m.retrieve("python"), Err(RetrievalError::Model(msg)) if msg.contains("reasoning")));
    // Conteúdo vazio vale como resposta (aqui, consulta vazia para a base).
    assert!(matches!(m.retrieve("python"), Err(RetrievalError::Input(_))));
}

#[test]
fn llm_config_validation() {
    let m = manager("hyde", json!({"base_url": "http://127.0.0.1:9/v1"}));
    assert!(matches!(m.retrieve("python"), Err(RetrievalError::InvalidConfig(msg)) if msg.contains("llm.model")));
}
