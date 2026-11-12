//! Contrato e comportamento sem modelo: as 44 estratégias com dependências falsas, erros,
//! parâmetros entregues ao provider e as extensões do port (recência com fuso, `parent_content`
//! do chunking, self-query nativo, busca por vetor).

mod common;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use chunking::ChunkManager;
use retriever::core::time::{now, parse_iso};
use retriever::{
    Document, EXTERNAL_PROVIDERS, LEGACY_RETRIEVAL_METHODS, RETRIEVER_CONTRACT, RetrievalError, RetrievalManager, Retriever,
    SUPPORTED_PROVIDERS, SUPPORTED_STRATEGIES, SearchRequest, SearchType,
};
use serde_json::{Value, json};

fn documents() -> Vec<Value> {
    vec![
        json!({"page_content": "Python possui tipagem dinâmica e uma sintaxe simples.", "metadata": {"id": "python", "kind": "code", "parent_id": "guide"}}),
        json!({"page_content": "Banana é uma fruta tropical amarela.", "metadata": {"id": "banana", "kind": "food"}}),
        json!({"page_content": "Busca híbrida combina recuperação densa e lexical.", "metadata": {"id": "search", "kind": "search"}}),
    ]
}

/// Configuração do teste do Python (`make_config`) com tudo injetado.
fn full_manager(strategy: &str) -> RetrievalManager {
    let cfg = json!({
        "strategy": strategy,
        "provider": "local",
        "documents": documents(),
        "k": 2,
        "fetch_k": 3,
        "top_n": 2,
        "options": {
            "parent_store": {"guide": {"page_content": "Guia completo de Python.", "metadata": {"id": "guide"}}},
            "strategies": ["dense", "bm25"],
            "stages": ["dense", "rerank"],
        },
    });
    let deps =
        ["embedder", "query_generator", "query_constructor", "compressor", "reranker", "neighbor_loader", "router", "custom", "provider"];
    common::inject(RetrievalManager::from_value(cfg).unwrap(), &deps, &documents())
}

#[test]
fn catalog() {
    assert_eq!(SUPPORTED_STRATEGIES.len(), 44);
    assert_eq!(SUPPORTED_PROVIDERS.len() + EXTERNAL_PROVIDERS.len(), 118);
    assert_eq!(RetrievalManager::strategies(), SUPPORTED_STRATEGIES);
    assert_eq!(RetrievalManager::providers(), SUPPORTED_PROVIDERS);
    assert_eq!(RETRIEVER_CONTRACT.len(), 9);
    assert_eq!(LEGACY_RETRIEVAL_METHODS.len(), 12);
    let unique: HashSet<_> = SUPPORTED_STRATEGIES.iter().chain(EXTERNAL_PROVIDERS).collect();
    assert_eq!(unique.len(), 44 + 114);
}

#[test]
fn every_strategy_runs_with_injected_dependencies() {
    let mut executed = HashSet::new();
    for &strategy in SUPPORTED_STRATEGIES {
        let m = full_manager(strategy);
        let result = m.retrieve("python").unwrap_or_else(|e| panic!("{strategy}: {e}"));
        assert!(!result.is_empty() && result.len() <= 2, "{strategy}: {} resultados", result.len());
        for (i, d) in result.iter().enumerate() {
            assert_eq!(d.metadata["retrieval_rank"], json!(i + 1), "{strategy}");
            assert!(d.metadata["retrieval_strategy"].is_string(), "{strategy}");
        }
        // O resto do contrato vem de `retrieve`.
        m.load_retriever().unwrap();
        assert_eq!(m.invoke("python").unwrap(), result, "{strategy}: invoke");
        assert_eq!(m.get_relevant_documents("python").unwrap(), result, "{strategy}");
        assert_eq!(m.batch(&["python", "python"]).unwrap(), vec![result.clone(), result.clone()]);
        assert_eq!(m.stream("python").map(Result::unwrap).collect::<Vec<_>>(), vec![result.clone()]);
        let mixed = m.batch_results(&["python", " "]);
        assert!(mixed[0].is_ok() && matches!(mixed[1], Err(RetrievalError::Input(_))), "{strategy}");
        executed.insert(strategy);
    }
    assert_eq!(executed.len(), 44);
}

#[test]
fn configuration_errors() {
    let err = |v: Value| RetrievalManager::from_value(v).err().unwrap_or_else(|| panic!("aceitou configuração inválida"));
    assert!(matches!(err(json!({"strategy": "unknown"})), RetrievalError::UnsupportedStrategy(..)));
    assert!(matches!(err(json!({"provider": "pinecone"})), RetrievalError::ExternalProvider(p) if p == "pinecone"));
    assert!(matches!(err(json!({"provider": "postgres"})), RetrievalError::ExternalProvider(p) if p == "pgvector"));
    assert!(matches!(err(json!({"provider": "nope"})), RetrievalError::UnsupportedProvider(..)));
    for bad in [json!({"k": 0}), json!({"fetch_k": false}), json!({"weights": [-0.1]}), json!({"score_threshold": "alto"}), json!({"db": {}})] {
        assert!(matches!(err(bad.clone()), RetrievalError::InvalidConfig(_)), "{bad}");
    }
    let m = full_manager("dense");
    assert!(matches!(m.retrieve("  "), Err(RetrievalError::Input(_))));
    // Fusão dentro de fusão e pesos que não batem com os componentes.
    let fusion = m.retrieve_with("python", json!({"strategy": "rrf", "components": ["dense", "hybrid"]}));
    assert!(matches!(fusion, Err(RetrievalError::InvalidConfig(_))));
    let weights = m.retrieve_with("python", json!({"strategy": "weighted_hybrid", "weights": [1.0]}));
    assert!(matches!(weights, Err(RetrievalError::InvalidConfig(_))));
}

#[test]
fn missing_dependencies() {
    let bare = |strategy: &str| RetrievalManager::from_value(json!({"strategy": strategy, "documents": documents()})).unwrap();
    for s in ["router", "custom"] {
        assert!(matches!(bare(s).retrieve("python"), Err(RetrievalError::InvalidConfig(_))), "{s}");
    }
    for s in ["dense", "multi_query", "hyde", "graph", "colbert_rerank"] {
        assert!(matches!(bare(s).retrieve("python"), Err(RetrievalError::MissingDependency(_))), "{s}");
    }
    let with_embedder = bare("llm_rerank").with_embedder(common::fake_embedder());
    assert!(matches!(with_embedder.retrieve("python"), Err(RetrievalError::MissingDependency(_))));
    // Lexicais não precisam de nada.
    for s in ["bm25", "bm25_plus", "tfidf", "keyword", "fuzzy", "metadata", "sparse", "recency"] {
        assert!(bare(s).retrieve("python").is_ok(), "{s}");
    }
}

#[test]
fn nested_strategies_are_bounded() {
    let looping = full_manager("router").with_router(|_, _| "router".into());
    assert!(matches!(looping.retrieve("python"), Err(RetrievalError::InvalidConfig(m)) if m.contains("nested")));
    let fallback = full_manager("fallback");
    match fallback.retrieve_with("python", json!({"strategies": ["fallback"]})) {
        Err(RetrievalError::AllFailed(m)) => assert!(m.contains("nested"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn fallback_reports_every_failure() {
    let m = RetrievalManager::from_value(json!({"strategy": "fallback", "documents": documents(), "strategies": ["custom", "router"]})).unwrap();
    match m.retrieve("python") {
        Err(RetrievalError::AllFailed(msg)) => assert!(msg.contains("custom strategy") && msg.contains("router strategy"), "{msg}"),
        other => panic!("{other:?}"),
    }
    // Resultado vazio também passa para a próxima.
    let m = full_manager("fallback");
    let r = m.retrieve_with("python", json!({"strategies": ["keyword", "bm25"], "filters": {"kind": "food"}})).unwrap();
    assert_eq!(r[0].metadata["retrieval_strategy"], "bm25");
}

/// Captura os pedidos feitos ao provider.
fn recording_provider() -> (Arc<Mutex<Vec<String>>>, Arc<dyn retriever::SearchProvider>) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let sink = log.clone();
    let docs: Vec<Document> = documents().into_iter().map(retriever::core::documents::as_document).collect();
    let provider = move |r: &SearchRequest| {
        sink.lock().unwrap().push(format!(
            "{}|{:?}|{}|{:?}|{}|{:?}|{}|{:?}",
            r.query,
            r.vector,
            r.k,
            r.fetch_k,
            r.search_type.as_str(),
            r.score_threshold,
            Value::Object(r.filters.clone()),
            r.collection
        ));
        Ok(docs.clone())
    };
    (log, Arc::new(provider))
}

#[test]
fn provider_receives_search_parameters() {
    let (log, provider) = recording_provider();
    let m = RetrievalManager::from_value(json!({
        "strategy": "similarity_threshold", "provider": "custom", "k": 2, "fetch_k": 8,
        "score_threshold": 0.75, "filters": {"kind": "code"}, "collection_name": "docs",
    }))
    .unwrap()
    .with_provider(provider);
    assert_eq!(m.retrieve("python").unwrap().len(), 2);
    m.retrieve_with("python", json!({"strategy": "dense_mmr"})).unwrap();
    m.retrieve_with("python", json!({"strategy": "graph"})).unwrap();
    m.retrieve_img("imagens", &[0.1, 0.2], 1).unwrap();
    let log = log.lock().unwrap();
    assert_eq!(log[0], r#"python|None|2|None|similarity_score_threshold|Some(0.75)|{"kind":"code"}|Some("docs")"#);
    assert_eq!(log[1], r#"python|None|2|Some(8)|mmr|Some(0.75)|{"kind":"code"}|Some("docs")"#);
    // Fontes especializadas não mandam limiar (como o `_provider_results` do Python).
    assert_eq!(log[2], r#"python|None|2|None|similarity|None|{"kind":"code"}|Some("docs")"#);
    assert_eq!(log[3], r#"|Some([0.1, 0.2])|1|None|similarity|None|{"kind":"code"}|Some("imagens")"#);
}

#[test]
fn injected_fusion_components() {
    let (log, provider) = recording_provider();
    let m = full_manager("ensemble").with_component(provider);
    let r = m.retrieve_with("python", json!({"components": ["bm25"], "weights": [1.0, 3.0]})).unwrap();
    // O componente injetado recebe `fetch_k` e, com peso maior, decide a ordem.
    assert!(log.lock().unwrap()[0].starts_with("python|None|3|"));
    assert_eq!(r[0].metadata["id"], "python");
    assert_eq!(r.len(), 2);
}

#[test]
fn overrides_do_not_mutate() {
    let m = full_manager("dense");
    let r = m.retrieve_with("python", json!({"strategy": "bm25", "k_neighbors": 1})).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].metadata["retrieval_strategy"], "bm25");
    assert_eq!(m.config().strategy, "dense");
    assert_eq!(m.config().k, 2);
    assert_eq!(m.config().documents.len(), 3);
    let other = m.retrieve_with("fruta", json!({"documents": [{"page_content": "fruta fresca", "metadata": {"id": "x"}}]})).unwrap();
    assert_eq!(other.len(), 1);
}

#[test]
fn recency_with_time_zones() {
    let t = now();
    let docs = json!([
        {"page_content": "dez dias", "metadata": {"id": "10d", "timestamp": t - 10.0 * 86_400.0}},
        {"page_content": "um dia", "metadata": {"id": "1d", "timestamp": t - 86_400.0}},
        {"page_content": "sem data", "metadata": {"id": "none"}},
        {"page_content": "zero", "metadata": {"id": "zero", "timestamp": 0}},
        {"page_content": "futuro", "metadata": {"id": "future", "timestamp": t + 86_400.0}},
    ]);
    let m = RetrievalManager::from_value(json!({"strategy": "recency", "documents": docs, "k": 5, "decay_rate": 0.1})).unwrap();
    let r = m.retrieve("qualquer").unwrap();
    let ids: Vec<&str> = r.iter().map(|d| d.metadata["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["future", "1d", "10d", "none", "zero"]);
    let score = |i: usize| r[i].metadata["retrieval_score"].as_f64().unwrap();
    assert!((score(0) - 1.0).abs() < 1e-9, "idade negativa conta como 0");
    assert!((score(1) - (-0.1f64).exp()).abs() < 1e-4);
    assert!((score(2) - (-1.0f64).exp()).abs() < 1e-4);
    assert_eq!((score(3), score(4)), (0.0, 0.0));
    // Data sem fuso é UTC (no Python, hora local).
    assert_eq!(parse_iso("2026-01-01T00:00:00"), parse_iso("2026-01-01T00:00:00+00:00"));
    assert_eq!(parse_iso("2026-01-01T00:00:00-03:00"), parse_iso("2026-01-01T03:00:00Z"));
    assert_eq!(parse_iso("ontem"), None);
}

#[test]
fn parent_document_from_chunking_parent_child() {
    let text = "Fortaleza tem praias urbanas e jangadas no fim da tarde. ".repeat(4)
        + &"Rust garante segurança de memória sem coletor de lixo. ".repeat(4);
    let chunker = ChunkManager::from_value(json!({
        "strategy": "parent_child", "chunk_size": 60, "chunk_overlap": 0,
        "options": {"parent_chunk_size": 240, "include_parent_content": true},
    }))
    .unwrap();
    let children = chunker.chunk_documents_with(&text, None).unwrap();
    let parents: HashSet<&str> = children.iter().map(|c| c.metadata["parent_content"].as_str().unwrap()).collect();
    assert!(children.len() > parents.len() && parents.len() >= 2);

    let docs: Vec<Value> = children.iter().map(|c| serde_json::to_value(c).unwrap()).collect();
    let m = RetrievalManager::from_value(json!({
        "strategy": "parent_document", "documents": docs, "k": 3, "base_strategy": "bm25",
    }))
    .unwrap();
    let r = m.retrieve("memória coletor de lixo Rust").unwrap();
    assert!(r[0].page_content.contains("Rust garante"), "{}", r[0].page_content);
    assert!(parents.contains(r[0].page_content.as_str()), "devolve o texto do pai inteiro");
    assert_eq!(r[0].metadata["id"], r[0].metadata["parent_id"]);
    assert!(!r[0].metadata.contains_key("parent_content") && !r[0].metadata.contains_key("child_index"));
    let unique: HashSet<&str> = r.iter().map(|d| d.page_content.as_str()).collect();
    assert_eq!(unique.len(), r.len(), "pais sem repetição");
}

#[test]
fn self_query_with_llm_and_metadata_fields() {
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let seen = prompts.clone();
    let llm = move |p: &str| -> retriever::Result<String> {
        seen.lock().unwrap().push(p.to_string());
        Ok(r#"Claro: {"query": "sintaxe", "filter": {"kind": "code", "inexistente": 1, "kind_obj": {"$eq": 1}}}"#.into())
    };
    let m = RetrievalManager::from_value(json!({
        "strategy": "self_query", "documents": documents(), "k": 3,
        "metadata_fields": [{"name": "kind", "description": "tipo do documento", "type": "string"}],
        "document_content_description": "trechos técnicos",
    }))
    .unwrap()
    .with_embedder(common::fake_embedder())
    .with_llm(Arc::new(llm));
    let r = m.retrieve("documentos de código sobre sintaxe").unwrap();
    assert_eq!(r.len(), 1, "só `kind` entra no filtro");
    assert_eq!(r[0].metadata["id"], "python");
    let prompt = &prompts.lock().unwrap()[0];
    assert!(prompt.contains("- kind (string): tipo do documento") && prompt.contains("trechos técnicos"));

    // Resposta sem JSON: a consulta vai inteira para a base.
    let m = RetrievalManager::from_value(json!({"strategy": "self_query", "documents": documents(), "metadata_fields": [{"name": "kind"}]}))
        .unwrap()
        .with_embedder(common::fake_embedder())
        .with_llm(Arc::new(|_: &str| Ok("não sei".to_string())));
    assert_eq!(m.retrieve("banana").unwrap()[0].metadata["id"], "banana");
}

#[test]
fn search_by_vector_and_multimodal_text() {
    let m = full_manager("dense");
    let python = common::VOCABULARY.iter().map(|w| if *w == "python" { 1.0 } else { 0.0 }).collect::<Vec<f32>>();
    let r = m.retrieve_by_vector(&python, Value::Null).unwrap();
    assert_eq!(r[0].metadata["id"], "python");
    assert_eq!(r[0].metadata["retrieval_strategy"], "multimodal");
    assert!(matches!(m.retrieve_by_vector(&[], Value::Null), Err(RetrievalError::Input(_))));
    // Documento com imagem exige embedder multimodal.
    let with_image = m.retrieve_with("python", json!({"strategy": "multimodal", "documents": [{"page_content": "foto", "metadata": {"image": "x.png"}}]}));
    assert!(matches!(with_image, Err(RetrievalError::InvalidConfig(_))));
    assert_eq!(m.dense_retrieval("python", 1, true).unwrap()[0].metadata["retrieval_strategy"], "multimodal");
}

#[test]
fn embedding_crate_strategies_without_models() {
    let m = RetrievalManager::from_value(json!({
        "strategy": "hybrid", "documents": documents(), "k": 2,
        "embedding": {"strategy": "hashing", "options": {"n_features": 256}},
    }))
    .unwrap();
    m.load_retriever().unwrap();
    let r = m.retrieve("fruta amarela").unwrap();
    assert_eq!(r[0].metadata["id"], "banana");
    let dense = m.retrieve_with("sintaxe simples", json!({"strategy": "dense"})).unwrap();
    assert_eq!(dense[0].metadata["id"], "python");
    // Configuração só com o nome da estratégia.
    let m = RetrievalManager::from_value(json!({"strategy": "dense", "documents": documents(), "embedding": "hashing"})).unwrap();
    assert_eq!(m.retrieve("busca densa").unwrap()[0].metadata["id"], "search");
}

#[test]
fn cascade_and_rerank_details() {
    let m = full_manager("cascade");
    // Estágio de fusão: o resultado anterior entra no RRF com o novo.
    let r = m.retrieve_with("python", json!({"stages": ["bm25", "keyword"]})).unwrap();
    assert!(r[0].metadata["retrieval_score"].as_f64().unwrap() > 0.0);
    // Reranker injetado vale também no `llm_rerank` (sem LLM).
    let r = m.retrieve_with("python", json!({"strategy": "llm_rerank", "top_n": 1})).unwrap();
    assert_eq!(r.len(), 1);
    // `rerank` com estratégia base composta cai para `dense`.
    let r = m.retrieve_with("python", json!({"strategy": "rerank", "base_strategy": "cascade"})).unwrap();
    assert_eq!(r[0].metadata["retrieval_strategy"], "dense");
}

#[test]
fn llm_scores_are_parsed_like_python() {
    let replies = ["Nota: 7,5", "-2", "sem número", "10/10"];
    let i = Arc::new(Mutex::new(0usize));
    let llm = move |_: &str| -> retriever::Result<String> {
        let mut n = i.lock().unwrap();
        *n += 1;
        Ok(replies[(*n - 1) % replies.len()].to_string())
    };
    let docs = json!([
        {"page_content": "a", "metadata": {"id": 1}}, {"page_content": "b", "metadata": {"id": 2}},
        {"page_content": "c", "metadata": {"id": 3}}, {"page_content": "d", "metadata": {"id": 4}},
    ]);
    let m = RetrievalManager::from_value(json!({"strategy": "llm_rerank", "documents": docs, "top_n": 4, "k": 4, "base_strategy": "metadata"}))
        .unwrap()
        .with_llm(Arc::new(llm));
    let r = m.retrieve("q").unwrap();
    let got: Vec<(i64, f64)> = r.iter().map(|d| (d.metadata["id"].as_i64().unwrap(), d.metadata["retrieval_score"].as_f64().unwrap())).collect();
    assert_eq!(got, [(4, 10.0), (1, 7.5), (3, 0.0), (2, -2.0)]);
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "multi_thread")]
async fn async_contract() {
    use retriever::asynch::{abatch, abatch_results, ainvoke, astream};

    let m: Arc<dyn Retriever> = Arc::new(full_manager("bm25"));
    let one = ainvoke(m.clone(), "python".into()).await.unwrap();
    assert_eq!(one[0].metadata["id"], "python");
    let batch = abatch(m.clone(), vec!["python".into(), "banana".into()]).await.unwrap();
    assert_eq!(batch[1][0].metadata["id"], "banana");
    assert!(abatch(m.clone(), vec!["python".into(), "".into()]).await.is_err());
    let mixed = abatch_results(m.clone(), vec!["".into(), "python".into()]).await;
    assert!(mixed[0].is_err() && mixed[1].is_ok());
    let mut rx = astream(m, "python".into());
    assert_eq!(rx.recv().await.unwrap().unwrap(), one);
    assert!(rx.recv().await.is_none());
}

#[test]
fn search_type_names() {
    assert_eq!(SearchType::Mmr.as_str(), "mmr");
}
