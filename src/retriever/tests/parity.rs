//! Paridade com o `RetrievalManager` do Python: `fixtures/parity.json` foi gerado por
//! `fixtures/gen_parity.py` com as mesmas dependências falsas de `common`. Compara conteúdo,
//! ordem e metadados (notas com tolerância de float32).

mod common;

use retriever::{Document, RetrievalManager, Retriever};
use serde_json::{Map, Value, json};

const FIXTURE: &str = include_str!("fixtures/parity.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("parity.json válido")
}

fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            (x - y).abs() <= 1e-5 * x.abs().max(1.0)
        }
        _ => a == b,
    }
}

fn compare(name: &str, expected: &[Value], actual: &[Document], rank_only: bool) {
    let ids = |docs: Vec<Option<&Value>>| docs.into_iter().map(|v| v.cloned().unwrap_or(Value::Null)).collect::<Vec<_>>();
    let exp_ids = ids(expected.iter().map(|e| e["metadata"].get("id")).collect());
    let act_ids = ids(actual.iter().map(|d| d.metadata.get("id")).collect());
    assert_eq!(exp_ids, act_ids, "{name}: ordem dos documentos");
    if rank_only {
        return;
    }
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(e["page_content"].as_str().unwrap(), a.page_content, "{name}[{i}]: conteúdo");
        let em = e["metadata"].as_object().unwrap();
        let mut ek: Vec<&String> = em.keys().collect();
        let mut ak: Vec<&String> = a.metadata.keys().collect();
        ek.sort();
        ak.sort();
        assert_eq!(ek, ak, "{name}[{i}]: chaves de metadado");
        for (k, v) in em {
            assert!(same_value(v, &a.metadata[k]), "{name}[{i}].{k}: Python {v} × Rust {}", a.metadata[k]);
        }
    }
}

fn config(f: &Value, strategy: &str, extra: &Map<String, Value>) -> Value {
    let mut cfg = f["base"].as_object().unwrap().clone();
    cfg.insert("documents".into(), f["documents"].clone());
    cfg.insert("strategy".into(), json!(strategy));
    let mut options = Map::new();
    for (k, v) in extra {
        if k == "options" {
            options.extend(v.as_object().unwrap().clone());
        } else {
            cfg.insert(k.clone(), v.clone());
        }
    }
    cfg.insert("options".into(), Value::Object(options));
    Value::Object(cfg)
}

fn deps(case: &Value) -> Vec<&str> {
    case["deps"].as_array().unwrap().iter().map(|d| d.as_str().unwrap()).collect()
}

#[test]
fn strategies_match_python() {
    let f = fixture();
    let documents = f["documents"].as_array().unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 62);
    let mut failed = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let run = std::panic::catch_unwind(|| {
            let cfg = config(&f, case["strategy"].as_str().unwrap(), case["config"].as_object().unwrap());
            let manager = common::inject(RetrievalManager::from_value(cfg).unwrap(), &deps(case), documents);
            let result = manager.retrieve(case["query"].as_str().unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
            compare(name, case["expected"].as_array().unwrap(), &result, case["rank_only"].as_bool().unwrap());
        });
        if run.is_err() {
            failed.push(name);
        }
    }
    assert!(failed.is_empty(), "casos diferentes do Python: {failed:?}");
}

#[test]
fn legacy_methods_match_python() {
    let f = fixture();
    let documents = f["documents"].as_array().unwrap();
    for case in f["legacy"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let m = common::inject(RetrievalManager::from_value(config(&f, "dense", &Map::new())).unwrap(), &deps(case), documents);
        let a = case["args"].as_array().unwrap();
        let q = a[0].as_str().unwrap();
        let n = |i: usize| a[i].as_u64().unwrap() as usize;
        let strs = |v: &Value| v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect::<Vec<_>>();
        let result = match case["method"].as_str().unwrap() {
            "tf_idf_retrieval" => m.tf_idf_retrieval(q, n(1)),
            "bm25_retrieval" => m.bm25_retrieval(q, n(1)),
            "dense_retrieval" => m.dense_retrieval(q, n(1), false),
            "hybrid_retrieval" if a.len() == 2 => m.hybrid_retrieval(q, n(1), None, "bm25"),
            "hybrid_retrieval" => {
                let w: Vec<f64> = a[2].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                m.hybrid_retrieval(q, n(1), Some(&w), a[3].as_str().unwrap())
            }
            "ensemble_retrieval" => {
                let funcs = strs(&a[1]);
                let refs: Vec<&str> = funcs.iter().map(String::as_str).collect();
                m.ensemble_retrieval(q, &refs, n(2), 10, 5, None)
            }
            "rerank_retrieval" => m.rerank_retrieval(q, n(1), n(2)),
            "contextual_compression_retrieval" => m.contextual_compression_retrieval(q, n(1)),
            "multi_query_retrieval" => m.multi_query_retrieval(q, n(1)),
            "rephrase_query_retrieval" => m.rephrase_query_retrieval(q, n(1)),
            "reprase_query_retrieval" => m.reprase_query_retrieval(q, n(1)),
            "self_query_retrieval" => m.self_query_retrieval(q, a[1].as_str().unwrap(), &[], &[], &[], n(5)),
            other => panic!("método desconhecido: {other}"),
        }
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        compare(name, case["expected"].as_array().unwrap(), &result, false);
    }
}
