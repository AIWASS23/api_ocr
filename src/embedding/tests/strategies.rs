//! Contrato de todas as estratégias (com um embedder injetado, como o teste do Python), as
//! estratégias só de lógica sem modelo (`tfidf`, `hashing`, `bm25`, `lsa`) e os erros.

use std::sync::Arc;

use embedding::{
    Embedder, EmbeddingConfig, EmbeddingError, EmbeddingManager, ImageInput, InputType, SUPPORTED_STRATEGIES, cosine, from_fn,
};
use serde_json::json;

const DOCS: [&str; 5] = [
    "Fortaleza possui uma extensa faixa litorânea, com praias urbanas e dunas.",
    "Embeddings representam textos como vetores numéricos em um espaço contínuo.",
    "Busca semântica aproxima consultas e documentos relacionados pelo significado.",
    "Rust é uma linguagem de sistemas rápida, segura e sem coletor de lixo.",
    "As praias do Ceará atraem turistas: Fortaleza, Jericoacoara e Canoa Quebrada.",
];

/// O `FakeUniversalEmbedder` do teste Python: `[len, query?, 2]` e imagens `[i, 1, 2]`.
struct Universal;

impl Embedder for Universal {
    fn model_name(&self) -> &str {
        "fake"
    }

    fn embed(&self, texts: &[&str], input_type: InputType) -> embedding::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| vec![t.chars().count() as f32, if input_type == InputType::Query { 1.0 } else { 0.0 }, 2.0]).collect())
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn embed_images(&self, images: &[ImageInput]) -> embedding::Result<Vec<Vec<f32>>> {
        Ok((0..images.len()).map(|i| vec![i as f32, 1.0, 2.0]).collect())
    }
}

fn manager(config: serde_json::Value) -> EmbeddingManager {
    EmbeddingManager::from_value(config).unwrap()
}

#[test]
fn every_strategy_implements_the_full_contract() {
    for strategy in SUPPORTED_STRATEGIES {
        let m = manager(json!({"strategy": strategy})).with_embedder(Arc::new(Universal));
        assert!(m.is_loaded(), "{strategy}");
        assert_eq!(m.embed(&DOCS[..3], InputType::Document).unwrap().len(), 3, "{strategy}");
        assert_eq!(m.embed_docs(&DOCS[..3]).unwrap().len(), 3);
        assert_eq!(m.embed_documents(&DOCS[..3]).unwrap()[0].len(), 3);
        assert_eq!(m.embed_query("consulta").unwrap(), vec![8.0, 1.0, 2.0]);
        assert_eq!(m.embed_images(&[ImageInput::from("x.png")]).unwrap(), vec![vec![0.0, 1.0, 2.0]]);
        assert_eq!(m.embed_sparse(&["ab"], InputType::Document).unwrap()[0].indices, vec![0, 2]);
        assert_eq!(m.dimension(), Some(3));
        assert!(m.embed(&[], InputType::Query).unwrap().is_empty());
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_contract() {
    use embedding::asynch::*;
    let m: Arc<dyn Embedder> = Arc::new(manager(json!({"strategy": "custom"})).with_embedder(Arc::new(Universal)));
    assert_eq!(aembed(m.clone(), vec!["abc".into()], InputType::Query).await.unwrap(), vec![vec![3.0, 1.0, 2.0]]);
    assert_eq!(aembed_documents(m.clone(), vec!["a".into(), "b".into()]).await.unwrap().len(), 2);
    assert_eq!(aembed_query(m.clone(), "ab".into()).await.unwrap(), vec![2.0, 1.0, 2.0]);
    assert_eq!(aembed_images(m, vec![ImageInput::from("x.png")]).await.unwrap().len(), 1);
}

#[test]
fn manager_finalizes_like_python() {
    // dimensions corta, normalize normaliza; vetores com NaN e linhas a menos são recusados.
    let m = manager(json!({"strategy": "custom", "dimensions": 2, "normalize": true})).with_embedder(Arc::new(Universal));
    let v = m.embed(&["abc"], InputType::Document).unwrap();
    assert_eq!(v[0].len(), 2);
    assert!((v[0][0] - 1.0).abs() < 1e-6, "[3, 0] normalizado");
    let bad = manager(json!({"strategy": "custom"})).with_embedder(from_fn("nan", |t, _| Ok(t.iter().map(|_| vec![f32::NAN]).collect())));
    assert!(bad.embed(&["x"], InputType::Document).unwrap_err().to_string().contains("NaN"));
    let short = manager(json!({"strategy": "custom"})).with_embedder(from_fn("short", |_, _| Ok(vec![vec![1.0]])));
    assert!(short.embed(&["a", "b"], InputType::Document).unwrap_err().to_string().contains("1 rows for 2"));
    // Texto só: imagens dão erro de capacidade.
    let text_only = manager(json!({"strategy": "custom"})).with_embedder(from_fn("t", |t, _| Ok(t.iter().map(|_| vec![1.0]).collect())));
    assert!(matches!(text_only.embed_images(&[ImageInput::from("x.png")]), Err(EmbeddingError::ImagesUnsupported(_))));
}

#[test]
fn configuration_errors() {
    assert!(matches!(EmbeddingManager::from_value(json!({"strategy": "openai"})), Err(EmbeddingError::PaidProvider(_))));
    assert!(matches!(EmbeddingManager::from_value(json!({"strategy": "voyage"})), Err(EmbeddingError::PaidProvider(_))));
    let custom = manager(json!({"strategy": "custom"}));
    assert!(custom.embed(&["x"], InputType::Document).unwrap_err().to_string().contains("with_embedder"));
    let bogus = manager(json!({"strategy": "tfidf", "options": {"bogus": 1}}));
    assert!(bogus.embed(&["x y"], InputType::Document).unwrap_err().to_string().contains("bogus"));
    assert!(EmbeddingManager::new(EmbeddingConfig::new("sentence_transformer").unwrap()).is_ok());
}

#[test]
fn tfidf_and_hashing_basics() {
    let tfidf = manager(json!({"strategy": "tfidf", "dimensions": 64, "normalize": true, "options": {"ngram_range": [1, 2]}}));
    assert!(tfidf.embed(&["busca"], InputType::Query).unwrap_err().to_string().contains("before embedding queries"));
    let docs = tfidf.embed(&DOCS, InputType::Document).unwrap();
    let query = tfidf.embed(&["busca por documentos"], InputType::Query).unwrap();
    assert_eq!(query[0].len(), docs[0].len());
    let hashing = manager(json!({"strategy": "hashing", "dimensions": 128, "normalize": true}));
    assert_eq!(hashing.embed(&["consulta independente"], InputType::Query).unwrap()[0].len(), 128);
    assert_eq!(hashing.embed(&DOCS, InputType::Document).unwrap().len(), DOCS.len());
}

fn best(query: &[f32], docs: &[Vec<f32>], score: fn(&[f32], &[f32]) -> f32) -> usize {
    (0..docs.len()).max_by(|&a, &b| score(query, &docs[a]).partial_cmp(&score(query, &docs[b])).unwrap()).unwrap()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn bm25_scores_are_dot_products() {
    let m = manager(json!({"strategy": "bm25", "options": {"strip_accents": "unicode"}}));
    assert!(m.embed(&["praias"], InputType::Query).is_err(), "consulta antes dos documentos");
    let docs = m.embed(&DOCS, InputType::Document).unwrap();
    // Sem stop words, "de" (raro neste acervo, e 2x no documento de Rust) pesaria mais que os
    // dois termos do assunto; por isso a consulta vai sem ele.
    let q = m.embed(&["praias Fortaleza"], InputType::Query).unwrap();
    // "praias" e "fortaleza" aparecem uma vez nos documentos 0 e 4, que têm o mesmo tamanho
    // (10 termos): empatam no topo; os outros não pontuam.
    let scores: Vec<f32> = docs.iter().map(|d| dot(&q[0], d)).collect();
    assert!(scores[0] > 0.0 && (scores[0] - scores[4]).abs() < 1e-6, "{scores:?}");
    assert!(scores[1] == 0.0 && scores[2] == 0.0 && scores[3] == 0.0, "{scores:?}");
    // Termo de um documento só: só ele pontua.
    let single = m.embed(&["rust"], InputType::Query).unwrap();
    let s3 = dot(&single[0], &docs[3]);
    assert!(s3 > 0.0 && docs.iter().enumerate().all(|(i, d)| i == 3 || dot(&single[0], d) == 0.0));
    // Esparso: só os termos do texto.
    let sparse = m.embed_sparse(&["praias de Fortaleza"], InputType::Query).unwrap();
    assert_eq!(sparse[0].indices.len(), 3, "praias, de, fortaleza");
    let with_de = m.embed(&["praias de Fortaleza"], InputType::Query).unwrap();
    assert_eq!(best(&with_de[0], &docs, dot), 3, "BM25 puro: o termo raro decide");
    assert!(sparse[0].values.iter().all(|&v| v == 1.0));
}

#[test]
fn bm25_formula() {
    // Dois documentos: "aa aa bb" e "bb". Consulta "aa": só o primeiro pontua.
    let m = manager(json!({"strategy": "bm25", "options": {"k1": 1.2, "b": 0.75}}));
    let docs = m.embed(&["aa aa bb", "bb"], InputType::Document).unwrap();
    let q = m.embed(&["aa"], InputType::Query).unwrap();
    let (n, df, tf, dl, avgdl, k1, b) = (2f64, 1f64, 2f64, 3f64, 2f64, 1.2f64, 0.75f64);
    let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
    let want = idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl));
    assert!((dot(&q[0], &docs[0]) as f64 - want).abs() < 1e-6, "{} != {want}", dot(&q[0], &docs[0]));
    assert_eq!(dot(&q[0], &docs[1]), 0.0);
}

#[test]
fn lsa_dense_vectors_group_related_texts() {
    let m = manager(json!({"strategy": "lsa", "dimensions": 3, "options": {"strip_accents": "unicode"}}));
    assert!(m.embed(&["praias"], InputType::Query).is_err());
    let docs = m.embed(&DOCS, InputType::Document).unwrap();
    assert!(docs.iter().all(|d| d.len() == 3), "denso, 3 dimensões");
    let q = m.embed(&["praias de Fortaleza no Ceará"], InputType::Query).unwrap();
    let ranked = best(&q[0], &docs, cosine);
    assert!(ranked == 0 || ranked == 4, "praias: {ranked}");
    // Determinístico: o mesmo acervo dá os mesmos vetores.
    let again = manager(json!({"strategy": "lsa", "dimensions": 3, "options": {"strip_accents": "unicode"}}));
    assert_eq!(again.embed(&DOCS, InputType::Document).unwrap(), docs);
    // Mais componentes que documentos: limitado ao acervo.
    let wide = manager(json!({"strategy": "lsa", "dimensions": 50}));
    assert_eq!(wide.embed(&DOCS, InputType::Document).unwrap()[0].len(), DOCS.len());
}
