//! Paridade com o `EmbeddingManager` do Python (chatbot_rag): `fixtures/parity.json` foi gerado
//! rodando o código original (scikit-learn) nas mesmas configurações e textos. TF-IDF e hashing
//! têm de bater exatamente (até o arredondamento de `float32`).

use std::path::PathBuf;

use embedding::{EmbeddingManager, InputType};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn expected(e: &Value) -> Vec<Vec<f32>> {
    let dim = e["dim"].as_u64().unwrap() as usize;
    e["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let mut v = vec![0f32; dim];
            for (i, x) in r["indices"].as_array().unwrap().iter().zip(r["values"].as_array().unwrap()) {
                v[i.as_u64().unwrap() as usize] = x.as_f64().unwrap() as f32;
            }
            v
        })
        .collect()
}

/// Roda os casos da fixture; devolve os nomes rodados.
fn run() -> Vec<String> {
    let data: Value = serde_json::from_str(&std::fs::read_to_string(fixtures().join("parity.json")).unwrap()).unwrap();
    let mut ran = Vec::new();
    let mut failures = Vec::new();
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let manager = EmbeddingManager::from_value(case["config"].clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (s, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let want = expected(&step["expected"]);
            let texts: Vec<&str> = step["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
            let kind: InputType = step["input_type"].as_str().unwrap().parse().unwrap();
            let got = manager.embed(&texts, kind).unwrap_or_else(|e| panic!("{name} passo {s}: {e}"));
            if got.len() != want.len() || got.iter().zip(&want).any(|(g, w)| g.len() != w.len()) {
                failures.push(format!("{name} passo {s}: forma {}x{} != {}x{}", got.len(), got.first().map_or(0, Vec::len), want.len(), want.first().map_or(0, Vec::len)));
                continue;
            }
            for (r, (g, w)) in got.iter().zip(&want).enumerate() {
                let diff = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                if diff > 1e-6 {
                    failures.push(format!("{name} passo {s} linha {r}: diferença máxima {diff:e}"));
                }
            }
        }
        ran.push(name.to_string());
    }
    assert!(failures.is_empty(), "{} divergência(s):\n{}", failures.len(), failures.join("\n"));
    ran
}

#[test]
fn logic_strategies_match_scikit_learn_exactly() {
    let ran = run();
    assert!(ran.iter().any(|n| n.starts_with("tfidf_")) && ran.iter().any(|n| n.starts_with("hashing_")), "{ran:?}");
}
