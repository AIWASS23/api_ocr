//! Estratégias com modelos de verdade, todos em Rust/ONNX Runtime, dentro do processo.
//! Os testes que baixam modelos ou dependem de um servidor (Ollama) são `#[ignore]`:
//!
//! ```bash
//! cargo test -p chunking --all-features --test models -- --include-ignored
//! ```

#![allow(unused_imports, dead_code)]

use std::sync::Arc;

use chunking::{ChunkManager, Chunker};
use serde_json::json;

fn minilm() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minilm").to_string_lossy().into_owned()
}

const MD: &str = "# Guia do Contribuinte\n\nEste guia explica os tributos.\n\n## IPTU\n\nO IPTU é cobrado anualmente e pode ser parcelado em até 10 vezes.\n\n| Faixa | Alíquota |\n|---|---|\n| 1 | 0,5% |\n| 2 | 0,8% |\n\n## ISS\n\nPrestadores de serviço recolhem ISS mensalmente.\n";

#[cfg(feature = "docling")]
#[test]
fn docling_hybrid_chunks_by_document_structure() {
    let m = ChunkManager::from_value(json!({"strategy": "docling_hybrid", "chunk_size": 64, "model_name": minilm()})).unwrap();
    m.load_model(None).unwrap();
    let docs = m.chunk_documents(MD).unwrap();
    assert!(docs.len() >= 3, "{docs:#?}");
    // Cada chunk leva o caminho de títulos (contextualize) e os metadados do docling.
    let iptu = docs.iter().find(|d| d.page_content.contains("cobrado anualmente")).expect("parágrafo do IPTU");
    assert!(iptu.page_content.starts_with("Guia do Contribuinte\nIPTU\n"), "{:?}", iptu.page_content);
    assert_eq!(iptu.metadata["headings"], json!(["Guia do Contribuinte", "IPTU"]));
    // A tabela inteira, serializada em triplas, num chunk.
    assert!(docs.iter().any(|d| d.page_content.contains("0,5%") && d.page_content.contains("0,8%")), "{docs:#?}");
}

#[cfg(feature = "embeddings")]
#[test]
#[ignore = "baixa o modelo de embeddings (~130 MB)"]
fn semantic_with_local_onnx_embeddings() {
    let text = "O IPTU é cobrado anualmente. O imposto incide sobre imóveis urbanos. O valor depende da área construída. \
                O time venceu o campeonato ontem. A torcida comemorou nas ruas. O técnico elogiou os jogadores.";
    let m = ChunkManager::from_value(json!({"strategy": "semantic", "chunk_size": 500, "model_name": "BAAI/bge-small-en-v1.5",
        "breakpoint_threshold_type": "percentile", "options": {"breakpoint_threshold_amount": 50}}))
    .unwrap();
    m.load_model(None).unwrap();
    let chunks = m.chunk(text).unwrap();
    assert!(chunks.len() >= 2, "{chunks:?}");
    assert_eq!(chunks.join(" "), text, "nenhum texto perdido");
}

#[cfg(feature = "onnx")]
#[test]
#[ignore = "baixa o modelo ONNX (~90 MB)"]
fn late_chunking_with_onnx_runtime() {
    let text = "Maria Silva foi eleita prefeita em 2020. Ela tomou posse em janeiro. Seu primeiro ato foi reformar escolas.";
    let m = ChunkManager::from_value(json!({"strategy": "late_chunking", "chunk_size": 60, "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "options": {"base_strategy": "sentence", "max_tokens": 256}}))
    .unwrap();
    m.load_model(None).unwrap();
    let docs = m.chunk_documents(text).unwrap();
    assert_eq!(docs.len(), 3);
    for d in &docs {
        let v: Vec<f32> = serde_json::from_value(d.metadata["embedding"].clone()).unwrap();
        assert_eq!(v.len(), 384);
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-4);
    }
    // O vetor do chunk depende do documento inteiro: o mesmo trecho sozinho dá outro vetor.
    let alone = m.chunk_documents("Ela tomou posse em janeiro.").unwrap();
    let a: Vec<f32> = serde_json::from_value(alone[0].metadata["embedding"].clone()).unwrap();
    let b: Vec<f32> = serde_json::from_value(docs[1].metadata["embedding"].clone()).unwrap();
    let cos: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert!(cos < 0.999, "o contexto do documento deveria mudar o vetor (cos={cos})");
}

#[cfg(feature = "llm")]
#[test]
#[ignore = "precisa de um Ollama local com qwen3.5:4b"]
fn contextual_retrieval_with_a_real_llm() {
    let m = ChunkManager::from_value(json!({"strategy": "contextual_retrieval", "chunk_size": 200,
        "options": {"base_strategy": "markdown_header", "llm_model": "qwen3.5:4b", "llm_base_url": "http://localhost:11434/v1",
                    "llm_timeout_secs": 300, "llm_max_tokens": 200, "llm_extra": {"reasoning_effort": "none"}}}))
    .unwrap();
    m.load_model(None).unwrap();
    let docs = m.chunk_documents(MD).unwrap();
    assert!(!docs.is_empty());
    for d in &docs {
        let ctx = d.metadata["context"].as_str().unwrap();
        assert!(!ctx.is_empty(), "contexto vazio");
        assert!(d.page_content.ends_with(d.metadata["original_chunk"].as_str().unwrap()));
        eprintln!("CONTEXTO: {ctx}\n");
    }
}
