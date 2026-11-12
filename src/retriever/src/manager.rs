//! Ponto de entrada único — o `RetrievalManager` do Python: configure uma vez e busque com
//! qualquer estratégia, trocando parâmetros por chamada. Objetos (embedder, LLM, reranker,
//! providers, roteador...) entram pelos métodos `with_*`; o que a configuração descreve
//! (`embedding`, `llm`, `rerank_model`) é carregado no primeiro uso e reaproveitado.

use std::sync::Arc;

use chunking::Document;
use embedding::Embedder;
use serde_json::{Map, Value, json};

use crate::config::{EXTERNAL_PROVIDERS, RetrievalConfig, SUPPORTED_PROVIDERS, SUPPORTED_STRATEGIES};
use crate::context::{Ctx, Env};
use crate::core::documents::annotate;
use crate::deps::{Compressor, Llm, QueryGenerator, Reranker, SearchProvider, SearchRequest, SearchType};
use crate::error::{RetrievalError, Result};
use crate::retriever::Retriever;
use crate::strategies::multimodal::score_corpus;

pub struct RetrievalManager {
    config: RetrievalConfig,
    env: Env,
}

impl RetrievalManager {
    pub fn new(config: RetrievalConfig) -> Self {
        Self { config, env: Env::default() }
    }

    /// Nome de estratégia, mapa (com as chaves legadas do Python) ou `null`.
    pub fn from_value(value: Value) -> Result<Self> {
        Ok(Self::new(RetrievalConfig::from_value(value)?))
    }

    pub fn config(&self) -> &RetrievalConfig {
        &self.config
    }

    /// As 44 estratégias.
    pub fn strategies() -> &'static [&'static str] {
        SUPPORTED_STRATEGIES
    }

    /// Providers deste crate (`local`, `vector_db`, `in_memory`, `custom`).
    pub fn providers() -> &'static [&'static str] {
        SUPPORTED_PROVIDERS
    }

    /// Integrações do catálogo do Python que dependem de um `SearchProvider` injetado.
    pub fn external_providers() -> &'static [&'static str] {
        EXTERNAL_PROVIDERS
    }

    // ------------------------------------------------------------------ injeção

    /// Embedder das estratégias densas (no lugar de `embedding` da configuração).
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.env.deps.embedder = Some(embedder);
        self
    }

    /// LLM das estratégias que reescrevem a consulta, comprimem ou reordenam.
    pub fn with_llm(mut self, llm: Arc<dyn Llm>) -> Self {
        self.env.deps.llm = Some(llm);
        self
    }

    /// Reranker das estratégias `rerank`, `cross_encoder_rerank`, `colbert_rerank` e
    /// `llm_rerank` (no lugar do cross-encoder de `rerank_model`).
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.env.deps.reranker = Some(reranker);
        self
    }

    pub fn with_compressor(mut self, compressor: Arc<dyn Compressor>) -> Self {
        self.env.deps.compressor = Some(compressor);
        self
    }

    /// Fonte de documentos (banco vetorial, busca gerenciada, web, grafo, SQL...).
    pub fn with_provider(mut self, provider: Arc<dyn SearchProvider>) -> Self {
        self.env.deps.provider = Some(provider);
        self
    }

    /// Componente extra das estratégias de fusão (`hybrid`, `rrf`, `ensemble`...).
    pub fn with_component(mut self, component: Arc<dyn SearchProvider>) -> Self {
        self.env.deps.components.push(component);
        self
    }

    /// Retriever da estratégia `custom`.
    pub fn with_custom(mut self, retriever: Arc<dyn SearchProvider>) -> Self {
        self.env.deps.custom = Some(retriever);
        self
    }

    /// Gerador de consultas (`multi_query`, `query_expansion`, `decomposition`,
    /// `rephrase_query`, `hyde`, `step_back`), no lugar do LLM.
    pub fn with_query_generator(mut self, generator: Arc<dyn QueryGenerator>) -> Self {
        self.env.deps.query_generator = Some(generator);
        self
    }

    /// Escolhe a estratégia de cada consulta (`router`, `adaptive`): recebe a consulta e
    /// `options.strategies`.
    pub fn with_router(mut self, router: impl Fn(&str, &[String]) -> String + Send + Sync + 'static) -> Self {
        self.env.deps.router = Some(Arc::new(router));
        self
    }

    /// Self-query: devolve a consulta reescrita e os filtros extraídos dela.
    pub fn with_query_constructor(
        mut self,
        constructor: impl Fn(&str) -> Result<(String, Map<String, Value>)> + Send + Sync + 'static,
    ) -> Self {
        self.env.deps.query_constructor = Some(Arc::new(constructor));
        self
    }

    /// Vizinhos de um trecho (`small_to_big`).
    pub fn with_neighbor_loader(mut self, loader: impl Fn(&Document) -> Vec<Document> + Send + Sync + 'static) -> Self {
        self.env.deps.neighbor_loader = Some(Arc::new(loader));
        self
    }

    // ------------------------------------------------------------------ busca

    fn run(&self, config: RetrievalConfig, query: &str) -> Result<Vec<Document>> {
        let strategy = config.strategy.clone();
        Ctx::new(config, &self.env).execute(query, &strategy)
    }

    /// Alterações só desta chamada (`retrieve(query, strategy=..., k=...)` do Python), com as
    /// chaves da configuração; as desconhecidas vão para `options`.
    pub fn retrieve_with(&self, query: &str, overrides: Value) -> Result<Vec<Document>> {
        self.run(self.config.with_overrides(overrides)?, query)
    }

    /// Busca por um vetor de consulta já calculado (o `retrieve_img` do Python, que passava o
    /// embedding ao banco): no acervo local, com o embedder multimodal para os documentos; senão,
    /// no provider (`SearchRequest::vector`).
    pub fn retrieve_by_vector(&self, vector: &[f32], overrides: Value) -> Result<Vec<Document>> {
        let mut overrides = match overrides {
            Value::Null => Map::new(),
            Value::Object(m) => m,
            _ => return Err(RetrievalError::config("overrides must be a mapping")),
        };
        overrides.insert("strategy".into(), Value::String("multimodal".into()));
        let config = self.config.with_overrides(Value::Object(overrides))?;
        let ctx = Ctx::new(config, &self.env);
        if vector.is_empty() {
            return Err(RetrievalError::Input("query vector must not be empty".into()));
        }
        let values = if !ctx.cfg.documents.is_empty() && ctx.embedder()?.is_some() {
            score_corpus(&ctx, vector)?
        } else {
            let provider = ctx.provider()?;
            let request = SearchRequest {
                query: "",
                vector: Some(vector),
                k: ctx.cfg.k,
                fetch_k: None,
                search_type: SearchType::Similarity,
                score_threshold: None,
                filters: &ctx.cfg.filters,
                collection: ctx.cfg.collection_name.as_deref(),
            };
            let mut docs = provider.search(&request)?;
            docs.truncate(ctx.cfg.k);
            docs
        };
        Ok(annotate(values, "multimodal", ctx.cfg.k))
    }

    // ------------------------------------------------------------------ métodos legados

    /// `dense_retrieval(query, k_neighbors, clip)`: `multimodal` com `clip`, senão `dense`.
    pub fn dense_retrieval(&self, query: &str, k_neighbors: usize, clip: bool) -> Result<Vec<Document>> {
        let strategy = if clip { "multimodal" } else { "dense" };
        self.retrieve_with(query, json!({"strategy": strategy, "k": k_neighbors}))
    }

    /// `retrieve_img(collection_name, query_embed, k_neighbors)`.
    pub fn retrieve_img(&self, collection_name: &str, query_embed: &[f32], k_neighbors: usize) -> Result<Vec<Document>> {
        self.retrieve_by_vector(query_embed, json!({"collection_name": collection_name, "k": k_neighbors}))
    }

    pub fn tf_idf_retrieval(&self, query: &str, k_neighbors: usize) -> Result<Vec<Document>> {
        self.retrieve_with(query, json!({"strategy": "tfidf", "k": k_neighbors}))
    }

    pub fn bm25_retrieval(&self, query: &str, k_neighbors: usize) -> Result<Vec<Document>> {
        self.retrieve_with(query, json!({"strategy": "bm25", "k": k_neighbors}))
    }

    /// `weighted_hybrid` de `dense` com `lexical_method` (`bm25`); pesos padrão `[0.5, 0.5]`.
    pub fn hybrid_retrieval(&self, query: &str, k_neighbors: usize, weights: Option<&[f64]>, lexical_method: &str) -> Result<Vec<Document>> {
        let weights = weights.filter(|w| !w.is_empty()).unwrap_or(&[0.5, 0.5]);
        self.retrieve_with(
            query,
            json!({"strategy": "weighted_hybrid", "k": k_neighbors, "weights": weights, "components": ["dense", lexical_method]}),
        )
    }

    /// `rerank` sobre `k_neighbors` candidatos, devolvendo `top_n_rerank`.
    pub fn rerank_retrieval(&self, query: &str, k_neighbors: usize, top_n_rerank: usize) -> Result<Vec<Document>> {
        self.retrieve_with(query, json!({"strategy": "rerank", "fetch_k": k_neighbors, "top_n": top_n_rerank, "k": top_n_rerank}))
    }

    pub fn contextual_compression_retrieval(&self, query: &str, k_neighbors: usize) -> Result<Vec<Document>> {
        self.retrieve_with(query, json!({"strategy": "contextual_compression", "k": k_neighbors, "top_n": k_neighbors}))
    }

    pub fn multi_query_retrieval(&self, query: &str, k_neighbors: usize) -> Result<Vec<Document>> {
        self.retrieve_with(query, json!({"strategy": "multi_query", "k": k_neighbors}))
    }

    /// `ensemble` dos retrievers nomeados (mais os injetados com `with_component`).
    pub fn ensemble_retrieval(
        &self,
        query: &str,
        retriever_funcs: &[&str],
        k_neighbors: usize,
        k_neighbors_rerank: usize,
        top_n_rerank: usize,
        weights: Option<&[f64]>,
    ) -> Result<Vec<Document>> {
        self.retrieve_with(
            query,
            json!({
                "strategy": "ensemble",
                "retrievers": retriever_funcs,
                "k": k_neighbors,
                "fetch_k": k_neighbors_rerank,
                "top_n": top_n_rerank,
                "weights": weights.unwrap_or(&[]),
            }),
        )
    }

    /// `self_query` com os campos de metadado em três listas paralelas (nome, descrição, tipo).
    pub fn self_query_retrieval(
        &self,
        query: &str,
        document_content_description: &str,
        names_metadata_fields: &[&str],
        description_metadata_fields: &[&str],
        type_metadata_fields: &[&str],
        k_neighbors: usize,
    ) -> Result<Vec<Document>> {
        let fields: Vec<Value> = names_metadata_fields
            .iter()
            .zip(description_metadata_fields)
            .zip(type_metadata_fields)
            .map(|((n, d), t)| json!({"name": n, "description": d, "type": t}))
            .collect();
        self.retrieve_with(
            query,
            json!({
                "strategy": "self_query",
                "k": k_neighbors,
                "document_content_description": document_content_description,
                "metadata_fields": fields,
            }),
        )
    }

    pub fn rephrase_query_retrieval(&self, query: &str, k_neighbors: usize) -> Result<Vec<Document>> {
        self.retrieve_with(query, json!({"strategy": "rephrase_query", "k": k_neighbors}))
    }

    /// Nome original (com o erro de grafia) do `rephrase_query_retrieval`.
    pub fn reprase_query_retrieval(&self, query: &str, k_neighbors: usize) -> Result<Vec<Document>> {
        self.rephrase_query_retrieval(query, k_neighbors)
    }
}

impl Retriever for RetrievalManager {
    fn retrieve(&self, query: &str) -> Result<Vec<Document>> {
        self.run(self.config.clone(), query)
    }

    /// Carrega o embedder de `embedding`, o cliente de `llm` e, nas estratégias de reranking
    /// sem reranker injetado, o cross-encoder de `rerank_model`.
    fn load_retriever(&self) -> Result<()> {
        let ctx = Ctx::new(self.config.clone(), &self.env);
        ctx.embedder()?;
        if self.env.deps.llm.is_none() && self.config.llm.is_some() {
            ctx.llm()?;
        }
        let reranks = matches!(self.config.strategy.as_str(), "rerank" | "cross_encoder_rerank")
            || (self.config.strategy == "cascade" && ctx.cfg.option_strings("stages").is_empty());
        if reranks && self.env.deps.reranker.is_none() {
            ctx.cross_encoder()?;
        }
        Ok(())
    }
}
