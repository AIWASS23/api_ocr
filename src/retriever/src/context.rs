//! Contexto de uma execução — o equivalente do `RetrievalStrategyExecutor` do Python: a
//! configuração desta chamada, o acervo (com filtros), embeddings (com cache), provider, LLM,
//! componentes da fusão e a estratégia base. Cada estratégia recebe um `Ctx` e pode executar
//! outras por [`Ctx::execute`].

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use chunking::Document;
use embedding::{Embedder, EmbeddingConfig, EmbeddingManager, InputType};
use serde_json::{Map, Value};

use crate::config::{FUSION_STRATEGIES, NON_BASE_STRATEGIES, RetrievalConfig, normalize_strategy};
use crate::core::documents::annotate;
use crate::core::filters;
use crate::core::rank::{rank_scores, reciprocal_rank_fusion};
use crate::core::vectors::{argsort_desc, cosine_scores, mmr_indices};
use crate::deps::{Deps, Llm, Reranker, SearchProvider, SearchRequest, SearchType};
use crate::error::{RetrievalError, Result};
use crate::strategies;

/// Estado compartilhado entre chamadas de um manager: dependências injetadas e o que é
/// carregado sob demanda a partir da configuração (embedder, LLM, cross-encoder), com cache.
#[derive(Default)]
pub struct Env {
    pub deps: Deps,
    embedders: Mutex<HashMap<String, Arc<EmbeddingManager>>>,
    llms: Mutex<HashMap<String, Arc<dyn Llm>>>,
    rerankers: Mutex<HashMap<String, Arc<dyn Reranker>>>,
    /// Vetores já calculados: (embedder, tipo, texto) → vetor.
    vectors: Mutex<HashMap<(usize, InputType, String), Vec<f32>>>,
}

/// Teto do cache de vetores (entradas); ao passar, o cache recomeça.
const VECTOR_CACHE_LIMIT: usize = 100_000;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Env {
    pub fn new(deps: Deps) -> Self {
        Self { deps, ..Default::default() }
    }
}

/// Um componente da fusão: estratégia por nome ou provider injetado.
pub enum Component {
    Strategy(String),
    Provider(Arc<dyn SearchProvider>),
}

/// Profundidade máxima de estratégias aninhadas (`router` escolhendo `router`, `fallback`
/// dentro de `fallback`...); no Python, um `RecursionError`.
const MAX_DEPTH: usize = 16;

pub struct Ctx<'a> {
    pub cfg: RetrievalConfig,
    pub env: &'a Env,
    corpus: RefCell<Option<Vec<Document>>>,
    depth: Cell<usize>,
}

impl<'a> Ctx<'a> {
    pub fn new(cfg: RetrievalConfig, env: &'a Env) -> Self {
        Self { cfg, env, corpus: RefCell::new(None), depth: Cell::new(0) }
    }

    /// Contexto com a configuração alterada (o Python muda `config.k`/`filters` e restaura).
    pub fn child(&self, f: impl FnOnce(&mut RetrievalConfig)) -> Ctx<'a> {
        let mut cfg = self.cfg.clone();
        f(&mut cfg);
        let child = Ctx::new(cfg, self.env);
        child.depth.set(self.depth.get());
        child
    }

    /// Executa uma estratégia: valida a consulta, despacha e anota posição/estratégia.
    pub fn execute(&self, query: &str, strategy: &str) -> Result<Vec<Document>> {
        let name = normalize_strategy(strategy)?;
        if name != "multimodal" && query.trim().is_empty() {
            return Err(RetrievalError::Input("query must be a non-empty string".into()));
        }
        let depth = self.depth.get();
        if depth >= MAX_DEPTH {
            return Err(RetrievalError::config(format!("retrieval strategies nested more than {MAX_DEPTH} levels deep")));
        }
        self.depth.set(depth + 1);
        let values = strategies::run(name, self, query);
        self.depth.set(depth);
        Ok(annotate(values?, name, self.cfg.k))
    }

    // ------------------------------------------------------------------ acervo

    /// Documentos da configuração ou do provider, com os filtros aplicados.
    pub fn corpus(&self) -> Result<Vec<Document>> {
        if let Some(c) = self.corpus.borrow().as_ref() {
            return Ok(c.clone());
        }
        let docs = if !self.cfg.documents.is_empty() {
            self.cfg.documents.clone()
        } else if let Some(p) = &self.env.deps.provider {
            p.documents(self.cfg.collection_name.as_deref())?.unwrap_or_default()
        } else {
            Vec::new()
        };
        let filtered = filters::apply(docs, &self.cfg.filters);
        *self.corpus.borrow_mut() = Some(filtered.clone());
        Ok(filtered)
    }

    // ------------------------------------------------------------------ embeddings

    /// Embedder injetado ou o da configuração `embedding` (carregado uma vez por manager).
    pub fn embedder(&self) -> Result<Option<Arc<dyn Embedder>>> {
        if let Some(e) = &self.env.deps.embedder {
            return Ok(Some(e.clone()));
        }
        let Some(cfg) = &self.cfg.embedding else { return Ok(None) };
        let key = cfg.to_string();
        let mut cache = lock(&self.env.embedders);
        if let Some(m) = cache.get(&key) {
            return Ok(Some(m.clone() as Arc<dyn Embedder>));
        }
        let value = match cfg {
            Value::String(s) => Value::Object(Map::from_iter([("strategy".to_string(), Value::String(s.clone()))])),
            other => other.clone(),
        };
        let manager = Arc::new(EmbeddingManager::new(EmbeddingConfig::from_value(value)?)?);
        cache.insert(key, manager.clone());
        Ok(Some(manager as Arc<dyn Embedder>))
    }

    /// Vetores de vários textos, com cache por (embedder, tipo, texto).
    pub fn embed(&self, embedder: &Arc<dyn Embedder>, texts: &[&str], kind: InputType) -> Result<Vec<Vec<f32>>> {
        let id = Arc::as_ptr(embedder) as *const () as usize;
        let mut out: Vec<Option<Vec<f32>>> = {
            let cache = lock(&self.env.vectors);
            texts.iter().map(|t| cache.get(&(id, kind, t.to_string())).cloned()).collect()
        };
        let missing: Vec<usize> = out.iter().enumerate().filter(|(_, v)| v.is_none()).map(|(i, _)| i).collect();
        if !missing.is_empty() {
            let batch: Vec<&str> = missing.iter().map(|&i| texts[i]).collect();
            let vectors = embedder.embed(&batch, kind)?;
            if vectors.len() != batch.len() {
                return Err(RetrievalError::model("embedder returned a different number of vectors"));
            }
            let mut cache = lock(&self.env.vectors);
            if cache.len() + vectors.len() > VECTOR_CACHE_LIMIT {
                cache.clear();
            }
            for (&i, v) in missing.iter().zip(vectors) {
                cache.insert((id, kind, texts[i].to_string()), v.clone());
                out[i] = Some(v);
            }
        }
        Ok(out.into_iter().map(Option::unwrap_or_default).collect())
    }

    /// Há como fazer busca densa (embedder ou provider)?
    pub fn has_dense_source(&self) -> bool {
        self.env.deps.embedder.is_some() || self.cfg.embedding.is_some() || self.env.deps.provider.is_some()
    }

    // ------------------------------------------------------------------ provider

    pub fn provider(&self) -> Result<Arc<dyn SearchProvider>> {
        if let Some(p) = &self.env.deps.provider {
            return Ok(p.clone());
        }
        Err(match self.cfg.provider.as_str() {
            "custom" => RetrievalError::missing("custom provider requires an injected SearchProvider (RetrievalManager::with_provider)"),
            p => RetrievalError::missing(format!("provider '{p}' requires documents (and an embedder for dense search) or an injected SearchProvider")),
        })
    }

    /// Busca no provider com `k`, filtros, limiar e, no MMR, `fetch_k`.
    pub fn provider_search(&self, query: &str, vector: Option<&[f32]>, search_type: SearchType) -> Result<Vec<Document>> {
        let provider = self.provider()?;
        let request = SearchRequest {
            query,
            vector,
            k: self.cfg.k,
            fetch_k: (search_type == SearchType::Mmr).then_some(self.cfg.fetch_k),
            search_type,
            score_threshold: self.cfg.score_threshold,
            filters: &self.cfg.filters,
            collection: self.cfg.collection_name.as_deref(),
        };
        let mut docs = provider.search(&request)?;
        docs.truncate(self.cfg.k);
        Ok(docs)
    }

    // ------------------------------------------------------------------ densa

    /// `_dense`: local (documentos + embedder) ou no provider.
    pub fn dense(&self, query: &str, search_type: SearchType) -> Result<Vec<Document>> {
        if !self.cfg.documents.is_empty()
            && let Some(embedder) = self.embedder()?
        {
            return self.local_dense(&embedder, query, search_type);
        }
        self.provider_search(query, None, search_type)
    }

    /// `_local_dense`: cosseno (ou MMR) sobre o acervo, com limiar de nota.
    pub fn local_dense(&self, embedder: &Arc<dyn Embedder>, query: &str, search_type: SearchType) -> Result<Vec<Document>> {
        let docs = self.corpus()?;
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
        let matrix = self.embed(embedder, &texts, InputType::Document)?;
        let q = self.embed(embedder, &[query], InputType::Query)?.remove(0);
        let scores = cosine_scores(&matrix, &q);
        let order = if search_type == SearchType::Mmr {
            mmr_indices(&matrix, &scores, self.cfg.k, self.cfg.option_f64("lambda_mult", 0.5) as f32)
        } else {
            argsort_desc(&scores)
        };
        let mut out = Vec::new();
        for i in order {
            let s = scores[i] as f64;
            if self.cfg.score_threshold.is_some_and(|t| s < t) {
                continue;
            }
            out.push(crate::core::documents::with_score(&docs[i], s));
            if out.len() >= self.cfg.k {
                break;
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------ LLM

    pub fn llm(&self) -> Result<Arc<dyn Llm>> {
        if let Some(l) = &self.env.deps.llm {
            return Ok(l.clone());
        }
        let Some(cfg) = &self.cfg.llm else {
            return Err(RetrievalError::missing("this retrieval strategy requires llm or a callback option"));
        };
        #[cfg(feature = "llm")]
        {
            let key = cfg.to_string();
            let mut cache = lock(&self.env.llms);
            if let Some(l) = cache.get(&key) {
                return Ok(l.clone());
            }
            let client: Arc<dyn Llm> = Arc::new(crate::core::llm::OpenAiCompatible::from_config(cfg)?);
            cache.insert(key, client.clone());
            Ok(client)
        }
        #[cfg(not(feature = "llm"))]
        {
            let _ = (cfg, &self.env.llms);
            Err(RetrievalError::FeatureDisabled { strategy: "llm", feature: "llm" })
        }
    }

    pub fn call_llm(&self, prompt: &str) -> Result<String> {
        self.llm()?.complete(prompt)
    }

    // ------------------------------------------------------------------ reranking

    /// Cross-encoder de `rerank_model` (carregado uma vez por modelo).
    pub fn cross_encoder(&self) -> Result<Arc<dyn Reranker>> {
        #[cfg(feature = "onnx")]
        {
            let name = self.cfg.rerank_model.clone();
            let mut cache = lock(&self.env.rerankers);
            if let Some(r) = cache.get(&name) {
                return Ok(r.clone());
            }
            let device = self.cfg.option_str("device").unwrap_or("cpu").to_string();
            let r: Arc<dyn Reranker> = Arc::new(crate::core::cross_encoder::CrossEncoder::load(&name, &device)?);
            cache.insert(name, r.clone());
            Ok(r)
        }
        #[cfg(not(feature = "onnx"))]
        {
            let _ = &self.env.rerankers;
            Err(RetrievalError::FeatureDisabled { strategy: "cross_encoder_rerank", feature: "onnx" })
        }
    }

    // ------------------------------------------------------------------ ordenação e fusão

    /// `_rank_scores` com o limite padrão `k`.
    pub fn rank(&self, docs: &[Document], scores: &[f64], positive_only: bool, limit: Option<usize>) -> Vec<Document> {
        rank_scores(docs, scores, positive_only, limit.unwrap_or(self.cfg.k))
    }

    /// Resultado de um componente: estratégia (sem fusão dentro de fusão) ou provider.
    pub fn component(&self, query: &str, component: &Component) -> Result<Vec<Document>> {
        match component {
            Component::Strategy(name) => {
                let selected = normalize_strategy(name)?;
                if FUSION_STRATEGIES.contains(&selected) {
                    return Err(RetrievalError::config("fusion components cannot recursively be fusion strategies"));
                }
                self.execute(query, selected)
            }
            Component::Provider(p) => {
                let request = SearchRequest {
                    query,
                    vector: None,
                    k: self.cfg.fetch_k,
                    fetch_k: None,
                    search_type: SearchType::Similarity,
                    score_threshold: self.cfg.score_threshold,
                    filters: &self.cfg.filters,
                    collection: self.cfg.collection_name.as_deref(),
                };
                let mut docs = p.search(&request)?;
                docs.truncate(self.cfg.fetch_k);
                Ok(docs)
            }
        }
    }

    /// Componentes da fusão: `options.components` ou `retrievers`, mais os injetados; padrão
    /// `dense` + `bm25`.
    pub fn components(&self) -> Vec<Component> {
        let mut names = self.cfg.option_strings("components");
        if names.is_empty() {
            names = self.cfg.retrievers.clone();
        }
        let mut out: Vec<Component> = names.into_iter().map(Component::Strategy).collect();
        out.extend(self.env.deps.components.iter().cloned().map(Component::Provider));
        if out.is_empty() {
            out = vec![Component::Strategy("dense".into()), Component::Strategy("bm25".into())];
        }
        out
    }

    /// `options.base_strategy` (padrão `dense`; estratégias compostas viram `dense`).
    pub fn base_strategy(&self) -> &'static str {
        match self.cfg.option_str("base_strategy").map(normalize_strategy) {
            Some(Ok(s)) if !NON_BASE_STRATEGIES.contains(&s) => s,
            _ => "dense",
        }
    }

    /// Candidatos da estratégia base, com `k = max(fetch_k, top_n)`.
    pub fn base_results(&self, query: &str) -> Result<Vec<Document>> {
        let k = self.cfg.fetch_k.max(self.cfg.top_n);
        let base = self.base_strategy();
        self.child(|c| c.k = k).component(query, &Component::Strategy(base.into()))
    }

    /// Fusão RRF sem pesos (`_fuse_rankings`).
    pub fn fuse(&self, rankings: &[Vec<Document>]) -> Vec<Document> {
        let weights = vec![1.0; rankings.len()];
        reciprocal_rank_fusion(rankings, &weights, self.cfg.option_f64("rrf_k", 60.0), self.cfg.k)
    }
}
