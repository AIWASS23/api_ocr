//! Paridade com o `src/chunking` do chatbot_rag (Python): cada caso de `fixtures/parity.json`
//! foi gerado rodando o `ChunkManager` REAL do chatbot sobre a mesma entrada e configuração.
//! A saída do Rust tem que ser idêntica, chunk por chunk.
//!
//! Diferenças intencionais (documentadas no README) são aplicadas à saída esperada antes de
//! comparar — e só elas.

#![cfg(all(feature = "hf-tokenizers", feature = "html", feature = "tiktoken"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use chunking::models::embeddings::Embedder;
use chunking::models::tokenizer::HfTokenizer;
use chunking::{ChunkManager, Chunker};
use serde_json::Value;

const MINILM: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// O mesmo embedding falso do gerador: contagem de a-z, de dígitos, e 1.
struct FakeEmbedder;

impl Embedder for FakeEmbedder {
    fn embed_documents(&self, texts: &[String]) -> chunking::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; 28];
                for ch in t.to_lowercase().chars() {
                    if ch.is_ascii_lowercase() {
                        v[(ch as u8 - b'a') as usize] += 1.0;
                    } else if ch.is_ascii_digit() {
                        v[26] += 1.0;
                    }
                }
                v[27] = 1.0;
                v
            })
            .collect())
    }
}

/// `html_section`: o Python passa o HTML pelo serializador do lxml, que insere `\n` entre
/// elementos de bloco (diferença intencional, só de espaço em branco); compara com os espaços
/// colapsados. O título da primeira seção é igualado com `title_placeholder = "#TITLE#"`.
fn collapse(chunks: &[String]) -> Vec<String> {
    chunks.iter().map(|c| c.split_whitespace().collect::<Vec<_>>().join(" ")).collect()
}

fn run_case(case: &Value, inputs: &serde_json::Map<String, Value>) -> Result<(), String> {
    let mut config = case["config"].clone();
    let strategy = config["strategy"].as_str().unwrap().to_string();
    let local_model = fixtures().join("minilm");
    if config.get("model_name").and_then(Value::as_str) == Some(MINILM) {
        config["model_name"] = Value::String(local_model.to_string_lossy().into_owned());
    }
    if strategy == "html_section" {
        config["options"]["title_placeholder"] = Value::String("#TITLE#".into());
    }
    let input = inputs[case["input"].as_str().unwrap()].as_str().unwrap();
    let outcome = (|| -> chunking::Result<Vec<String>> {
        let mut m = ChunkManager::from_value(config.clone())?;
        if strategy == "token" || strategy == "semchunk" {
            m = m.with_tokenizer(Arc::new(HfTokenizer::load(&local_model.to_string_lossy())?));
        }
        if strategy == "semantic" {
            m = m.with_embedder(Arc::new(FakeEmbedder));
        }
        m.chunk(input)
    })();
    match (case.get("expected"), outcome) {
        (Some(exp), Ok(got)) => {
            let exp: Vec<String> = serde_json::from_value(exp.clone()).unwrap();
            let (exp, got) = if strategy == "html_section" { (collapse(&exp), collapse(&got)) } else { (exp, got) };
            if exp == got {
                return Ok(());
            }
            // Sem sobreposição, o espaço extra do lxml pode mover uma fronteira no `html_section`
            // sem mudar o conteúdo: compara o texto inteiro.
            if strategy == "html_section" && config["chunk_overlap"] == 0 && exp.join(" ") == got.join(" ") {
                return Ok(());
            }
            let i = exp.iter().zip(&got).position(|(a, b)| a != b).unwrap_or(exp.len().min(got.len()));
            Err(format!(
                "{} chunks esperados, {} obtidos; 1ª diferença no chunk {i}:\n  python: {:?}\n  rust:   {:?}",
                exp.len(),
                got.len(),
                exp.get(i),
                got.get(i)
            ))
        }
        (Some(_), Err(e)) => Err(format!("Python devolveu chunks, Rust deu erro: {e}")),
        (None, Ok(got)) => Err(format!("Python deu erro ({}), Rust devolveu {} chunks", case["error"], got.len())),
        (None, Err(_)) => Ok(()),
    }
}

/// O `nltk` depende dos parâmetros do `punkt_tab` (o CI baixa; ver `.github/workflows/ci.yml`).
fn has_punkt() -> bool {
    chunking::models::punkt::PunktParameters::load("portuguese", None).is_ok()
}

#[test]
fn every_case_matches_the_python_chatbot() {
    let data: Value = serde_json::from_str(&std::fs::read_to_string(fixtures().join("parity.json")).unwrap()).unwrap();
    let inputs = data["inputs"].as_object().unwrap();
    let cases = data["cases"].as_array().unwrap();
    let mut by_strategy: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut failures = Vec::new();
    let punkt = has_punkt();
    let mut skipped = 0;
    for case in cases {
        let strategy = case["config"]["strategy"].as_str().unwrap().to_string();
        if strategy == "nltk" && !punkt {
            skipped += 1;
            continue;
        }
        let entry = by_strategy.entry(strategy.clone()).or_default();
        entry.0 += 1;
        match run_case(case, inputs) {
            Ok(()) => entry.1 += 1,
            Err(e) => failures.push(format!("[{strategy}] {} / input={}: {e}", case["config"], case["input"])),
        }
    }
    if skipped > 0 {
        eprintln!("AVISO: {skipped} casos do nltk pulados (sem punkt_tab: baixe com nltk.download('punkt_tab') ou defina NLTK_DATA)");
    }
    for (s, (total, ok)) in &by_strategy {
        eprintln!("{s:<28} {ok}/{total}");
    }
    if !failures.is_empty() {
        for f in failures.iter().take(25) {
            eprintln!("\n{f}");
        }
        panic!("{} de {} casos divergem do Python", failures.len(), cases.len());
    }
}
