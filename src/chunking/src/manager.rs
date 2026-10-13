//! `ChunkManager` — uma configuração e um método para qualquer estratégia (o equivalente de
//! `chunk_manager.py`).
//!
//! ```no_run
//! use chunking::{ChunkConfig, ChunkManager, Chunker};
//! use serde_json::json;
//!
//! let m = ChunkManager::from_value(json!({"strategy": "token", "chunk_size": 256,
//!     "chunk_overlap": 64, "model_name": "BAAI/bge-m3"}))?;
//! m.load_model(None)?;                       // tokenizador do Hub (ou caminho local)
//! let chunks = m.chunk("texto...")?;
//! let por_frase = m.chunk_with("texto...", Some("sentence"))?;
//! # Ok::<(), chunking::ChunkError>(())
//! ```
//!
//! Modelos (tokenizador, embeddings, LLM, embeddings por token) são carregados sob demanda por
//! [`ChunkManager::load_model`] ou injetados (`with_*`). Um splitter externo pode substituir
//! todas as estratégias com [`ChunkManager::with_chunker`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use serde_json::{Map, Value};

use crate::chunker::{Chunker, Sizing};
use crate::config::{ChunkConfig, SUPPORTED_STRATEGIES, normalize_strategy};
use crate::core::options::Options;
use crate::document::Document;
use crate::error::{ChunkError, Result};
use crate::models::embeddings::Embedder;
use crate::models::llm::ContextModel;
use crate::models::token_embeddings::TokenEmbedder;
use crate::models::tokenizer::TextTokenizer;
use crate::strategies as s;

/// Opções consumidas pelo manager (configuram modelos ou a estratégia base), não pela estratégia.
const MANAGER_OPTIONS: &[&str] = &[
    "base_strategy",
    "base_options",
    "llm_base_url",
    "llm_model",
    "llm_api_key",
    "llm_api_key_env",
    "llm_timeout_secs",
    "llm_prompt",
    "llm_max_tokens",
    "llm_extra",
    "embedding_model",
    "max_tokens",
];

type Slot<T> = RwLock<Option<Arc<T>>>;

pub struct ChunkManager {
    config: ChunkConfig,
    tokenizer: Slot<dyn TextTokenizer>,
    embedder: Slot<dyn Embedder>,
    context_model: Slot<dyn ContextModel>,
    token_embedder: Slot<dyn TokenEmbedder>,
    external: Option<Arc<dyn Chunker>>,
    cache: Mutex<HashMap<&'static str, Arc<dyn Chunker>>>,
}

fn get<T: ?Sized>(slot: &Slot<T>) -> Option<Arc<T>> {
    slot.read().unwrap_or_else(PoisonError::into_inner).clone()
}

fn set<T: ?Sized>(slot: &Slot<T>, value: Arc<T>) {
    *slot.write().unwrap_or_else(PoisonError::into_inner) = Some(value);
}

impl ChunkManager {
    pub fn new(config: ChunkConfig) -> Result<Self> {
        // Revalida (aliases, limites), mesmo vindo de um `ChunkConfig` construído à mão.
        let config = ChunkConfig::from_value(serde_json::to_value(&config).map_err(|e| ChunkError::Json(e.to_string()))?)?;
        Ok(Self {
            config,
            tokenizer: RwLock::new(None),
            embedder: RwLock::new(None),
            context_model: RwLock::new(None),
            token_embedder: RwLock::new(None),
            external: None,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// A partir do mapa do YAML (`indexing.chunking` do chatbot).
    pub fn from_value(value: Value) -> Result<Self> {
        Self::new(ChunkConfig::from_value(value)?)
    }

    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }

    /// Nomes canônicos aceitos.
    pub fn strategies() -> &'static [&'static str] {
        SUPPORTED_STRATEGIES
    }

    pub fn requires_model(&self, strategy: Option<&str>) -> Result<bool> {
        crate::config::requires_model(strategy.unwrap_or(&self.config.strategy))
    }

    // ------------------------------------------------------------------ injeção de modelos

    pub fn with_tokenizer(self, tokenizer: Arc<dyn TextTokenizer>) -> Self {
        set(&self.tokenizer, tokenizer);
        self.clear_cache();
        self
    }

    pub fn with_embedder(self, embedder: Arc<dyn Embedder>) -> Self {
        set(&self.embedder, embedder);
        self.clear_cache();
        self
    }

    pub fn with_context_model(self, model: Arc<dyn ContextModel>) -> Self {
        set(&self.context_model, model);
        self.clear_cache();
        self
    }

    pub fn with_token_embedder(self, embedder: Arc<dyn TokenEmbedder>) -> Self {
        set(&self.token_embedder, embedder);
        self.clear_cache();
        self
    }

    /// Splitter externo no lugar das estratégias nativas (`options.chunker` do Python).
    pub fn with_chunker(mut self, chunker: Arc<dyn Chunker>) -> Self {
        self.external = Some(chunker);
        self
    }

    fn clear_cache(&self) {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner).clear();
    }

    /// Carrega só o modelo que a estratégia (ou a configurada) precisa. Sem modelo: não faz nada.
    pub fn load_model(&self, strategy: Option<&str>) -> Result<()> {
        let strategy = normalize_strategy(strategy.unwrap_or(&self.config.strategy))?;
        let opt_str = |k: &str| self.config.options.get(k).and_then(Value::as_str).map(str::to_string);
        let result = match strategy {
            "token" | "semchunk" | "docling_hybrid" if get(&self.tokenizer).is_none() => self.load_tokenizer(),
            "semantic" if get(&self.embedder).is_none() => self.load_embedder(opt_str("embedding_model")),
            "late_chunking" if get(&self.token_embedder).is_none() => self.load_token_embedder(opt_str("embedding_model")),
            "contextual_retrieval" if get(&self.context_model).is_none() => self.load_context_model(),
            _ => Ok(()),
        };
        self.clear_cache();
        result.map_err(|e| match e {
            ChunkError::Model(m) => ChunkError::Model(format!("Erro ao carregar modelo '{}': {m}", self.config.model_name)),
            other => other,
        })
    }

    #[cfg(feature = "hf-tokenizers")]
    fn load_tokenizer(&self) -> Result<()> {
        set(&self.tokenizer, Arc::new(crate::models::tokenizer::HfTokenizer::load(&self.config.model_name)?) as Arc<dyn TextTokenizer>);
        Ok(())
    }

    #[cfg(not(feature = "hf-tokenizers"))]
    fn load_tokenizer(&self) -> Result<()> {
        Err(ChunkError::FeatureDisabled { strategy: "token", feature: "hf-tokenizers" })
    }

    #[cfg(feature = "embeddings")]
    fn load_embedder(&self, model: Option<String>) -> Result<()> {
        let name = model.unwrap_or_else(|| self.config.model_name.clone());
        set(&self.embedder, Arc::new(crate::models::embeddings::FastEmbedder::load(&name)?) as Arc<dyn Embedder>);
        Ok(())
    }

    #[cfg(not(feature = "embeddings"))]
    fn load_embedder(&self, _model: Option<String>) -> Result<()> {
        Err(ChunkError::FeatureDisabled { strategy: "semantic", feature: "embeddings" })
    }

    #[cfg(feature = "onnx")]
    fn load_token_embedder(&self, model: Option<String>) -> Result<()> {
        let name = model.unwrap_or_else(|| self.config.model_name.clone());
        let max = self.config.options.get("max_tokens").and_then(Value::as_u64).unwrap_or(8192) as usize;
        let e = crate::models::token_embeddings::OnnxTokenEmbedder::load(&name, max)?;
        set(&self.token_embedder, Arc::new(e) as Arc<dyn TokenEmbedder>);
        Ok(())
    }

    #[cfg(not(feature = "onnx"))]
    fn load_token_embedder(&self, _model: Option<String>) -> Result<()> {
        Err(ChunkError::FeatureDisabled { strategy: "late_chunking", feature: "onnx" })
    }

    #[cfg(feature = "llm")]
    fn load_context_model(&self) -> Result<()> {
        let o = &self.config.options;
        let s = |k: &str| o.get(k).and_then(Value::as_str);
        let model = s("llm_model").ok_or_else(|| ChunkError::config("contextual_retrieval: options.llm_model é obrigatório"))?;
        let base = s("llm_base_url").unwrap_or("http://localhost:11434/v1");
        let key = s("llm_api_key")
            .map(str::to_string)
            .or_else(|| std::env::var(s("llm_api_key_env").unwrap_or("OPENAI_API_KEY")).ok());
        let timeout = std::time::Duration::from_secs(o.get("llm_timeout_secs").and_then(Value::as_u64).unwrap_or(60));
        let mut client = crate::models::llm::OpenAiCompatible::new(base, model, key, timeout)?;
        if let Some(p) = s("llm_prompt") {
            client.prompt = p.to_string();
        }
        if let Some(n) = o.get("llm_max_tokens").and_then(Value::as_u64) {
            client.max_tokens = n as u32;
        }
        if let Some(extra) = o.get("llm_extra").and_then(Value::as_object) {
            client.extra = extra.clone();
        }
        set(&self.context_model, Arc::new(client) as Arc<dyn ContextModel>);
        Ok(())
    }

    #[cfg(not(feature = "llm"))]
    fn load_context_model(&self) -> Result<()> {
        Err(ChunkError::FeatureDisabled { strategy: "contextual_retrieval", feature: "llm" })
    }

    fn need<T: ?Sized>(&self, slot: &Slot<T>, what: &str, strategy: &str) -> Result<Arc<T>> {
        get(slot).ok_or_else(|| {
            ChunkError::ModelNotLoaded(format!("{what} not loaded. Call load_model('{strategy}') first or inject one."))
        })
    }

    // ------------------------------------------------------------------ construção

    fn options_for(&self, strategy: &'static str, raw: &Map<String, Value>) -> Options {
        let mut m = raw.clone();
        for k in MANAGER_OPTIONS {
            m.remove(*k);
        }
        Options::new(strategy, &m)
    }

    /// Constrói (ou devolve do cache) a estratégia.
    pub fn build(&self, strategy: Option<&str>) -> Result<Arc<dyn Chunker>> {
        if let Some(ext) = &self.external {
            return Ok(ext.clone());
        }
        let name = normalize_strategy(strategy.unwrap_or(&self.config.strategy))?;
        if let Some(c) = self.cache.lock().unwrap_or_else(PoisonError::into_inner).get(name) {
            return Ok(c.clone());
        }
        let built: Arc<dyn Chunker> = Arc::from(self.construct(name, &self.config.options.clone())?);
        self.cache.lock().unwrap_or_else(PoisonError::into_inner).insert(name, built.clone());
        Ok(built)
    }

    fn construct(&self, name: &'static str, raw: &Map<String, Value>) -> Result<Box<dyn Chunker>> {
        let cfg = &self.config;
        let sizing = Sizing::new(cfg.chunk_size, cfg.chunk_overlap)?;
        let mut o = self.options_for(name, raw);
        let lang_opt = |o: &mut Options, default: &str| -> Result<String> {
            Ok(o.string("language")?.or_else(|| cfg.language.clone()).unwrap_or_else(|| default.to_string()))
        };
        Ok(match name {
            "character" => Box::new(s::character::ChunkCharacter::from_options(sizing, o)?),
            "recursive" => Box::new(s::recursive::ChunkRecursive::from_options(sizing, cfg.language.as_deref(), o)?),
            "word" => Box::new(s::word::ChunkWord::from_options(sizing, o)?),
            "sentence" => Box::new(s::sentence::ChunkSentence::from_options(sizing, o)?),
            "paragraph" => Box::new(s::paragraph::ChunkParagraph::from_options(sizing, o)?),
            "line" => Box::new(s::line::ChunkLine::from_options(sizing, o)?),
            "token" => Box::new(s::token::ChunkToken::new(self.need(&self.tokenizer, "Tokenizer", "token")?, sizing)?),
            "token_tiktoken" => Box::new(s::token_tiktoken::ChunkTikToken::from_options(sizing, o)?),
            "token_sentence_transformer" => self.sentence_transformer(sizing, o)?,
            "semchunk" => Box::new(s::semchunk::ChunkSemchunk::new(self.need(&self.tokenizer, "Tokenizer", "semchunk")?, sizing, o)?),
            "semantic" => Box::new(s::semantic::ChunkSemantic::new(
                self.need(&self.embedder, "Embeddings", "semantic")?,
                sizing,
                &cfg.breakpoint_threshold_type,
                o,
            )?),
            "nltk" => {
                let language = lang_opt(&mut o, "portuguese")?;
                Box::new(s::nltk::ChunkNltk::new(sizing, &language, o)?)
            }
            "spacy" => {
                let pipeline = o.string("pipeline")?.unwrap_or_else(|| "sentencizer".into());
                let language = lang_opt(&mut o, "pt")?;
                Box::new(s::spacy::ChunkSpacy::new(sizing, &pipeline, &language, o)?)
            }
            "markdown" => Box::new(s::markdown::ChunkMarkdown::new(sizing, o)?),
            "markdown_header" => Box::new(s::markdown_header::ChunkMarkdownHeader::new(sizing, o)?),
            "markdown_syntax" => Box::new(s::markdown_syntax::ChunkMarkdownSyntax::new(sizing, o)?),
            "html" => Box::new(s::html::chunk_html(sizing, o)?),
            "html_header" | "html_section" | "html_semantic" => self.html(name, sizing, o)?,
            "json" => Box::new(s::json::ChunkJson::new(sizing, o)?),
            "latex" => Box::new(s::latex::ChunkLatex::new(sizing, o)?),
            "code" => {
                let language = lang_opt(&mut o, "python")?;
                Box::new(s::code::ChunkCode::new(sizing, &language, "code", o)?)
            }
            "python" => Box::new(s::python::chunk_python(sizing, o)?),
            "javascript" => Box::new(s::javascript::ChunkJavaScript::new(sizing, o)?),
            "markdown_header_contextual" => Box::new(s::markdown_header_contextual::ChunkMarkdownHeaderContextual::new(sizing, o)?),
            "docling_hybrid" => Box::new(s::docling_hybrid::ChunkDoclingHybrid::new(
                sizing,
                self.need(&self.tokenizer, "Tokenizer", "docling_hybrid")?,
                cfg.document_type.as_deref(),
                o,
            )?),
            "table" => Box::new(s::table::ChunkTable::new(sizing, o)?),
            "parent_child" => Box::new(s::parent_child::ChunkParentChild::new(sizing, o)?),
            "legal" => Box::new(s::legal::ChunkLegal::new(sizing, o)?),
            "qa" => Box::new(s::qa::ChunkQa::new(sizing, o)?),
            "contextual_retrieval" => {
                let base = self.base(raw, "recursive", &["contextual_retrieval", "late_chunking"])?;
                let model = self.need(&self.context_model, "Context model (LLM)", "contextual_retrieval")?;
                Box::new(s::contextual_retrieval::ChunkContextualRetrieval::new(sizing, base, model, o)?)
            }
            "late_chunking" => {
                let base = self.base(raw, "sentence", &["contextual_retrieval", "late_chunking"])?;
                let embedder = self.need(&self.token_embedder, "Token embedder", "late_chunking")?;
                Box::new(s::late_chunking::ChunkLateChunking::new(sizing, base, embedder, o)?)
            }
            other => {
                return Err(ChunkError::UnsupportedStrategy(other.into(), SUPPORTED_STRATEGIES.join(", ")));
            }
        })
    }

    /// Estratégia base de `contextual_retrieval`/`late_chunking` (`base_strategy` + `base_options`).
    fn base(&self, raw: &Map<String, Value>, default: &str, forbidden: &[&str]) -> Result<Box<dyn Chunker>> {
        let name = normalize_strategy(raw.get("base_strategy").and_then(Value::as_str).unwrap_or(default))?;
        if forbidden.contains(&name) {
            return Err(ChunkError::config(format!("base_strategy '{name}' não pode ser usada como base")));
        }
        let opts = raw.get("base_options").and_then(Value::as_object).cloned().unwrap_or_default();
        self.construct(name, &opts)
    }

    #[cfg(feature = "html")]
    fn html(&self, name: &'static str, sizing: Sizing, o: Options) -> Result<Box<dyn Chunker>> {
        Ok(match name {
            "html_header" => Box::new(s::html_header::ChunkHtmlHeader::new(sizing, o)?),
            "html_section" => Box::new(s::html_section::ChunkHtmlSection::new(sizing, o)?),
            _ => Box::new(s::html_semantic::ChunkHtmlSemantic::new(sizing, o)?),
        })
    }

    #[cfg(not(feature = "html"))]
    fn html(&self, name: &'static str, _sizing: Sizing, _o: Options) -> Result<Box<dyn Chunker>> {
        Err(ChunkError::FeatureDisabled { strategy: name, feature: "html" })
    }

    #[cfg(feature = "hf-tokenizers")]
    fn sentence_transformer(&self, sizing: Sizing, mut o: Options) -> Result<Box<dyn Chunker>> {
        let model = o.string("model_name")?.unwrap_or_else(|| self.config.model_name.clone());
        o.ignore(&["model_kwargs"]);
        o.finish()?;
        let tok: Arc<dyn TextTokenizer> = Arc::new(crate::models::tokenizer::HfTokenizer::load(&model)?);
        Ok(Box::new(s::token_sentence_transformer::ChunkSentenceTransformerToken::new(tok, &model, sizing)?))
    }

    #[cfg(not(feature = "hf-tokenizers"))]
    fn sentence_transformer(&self, _sizing: Sizing, _o: Options) -> Result<Box<dyn Chunker>> {
        Err(ChunkError::FeatureDisabled { strategy: "token_sentence_transformer", feature: "hf-tokenizers" })
    }

    // ------------------------------------------------------------------ uso

    /// Divide com a estratégia configurada ou com outra só nesta chamada.
    pub fn chunk_with(&self, text: &str, strategy: Option<&str>) -> Result<Vec<String>> {
        self.build(strategy)?.chunk(text)
    }

    pub fn chunk_documents_with(&self, text: &str, strategy: Option<&str>) -> Result<Vec<Document>> {
        self.build(strategy)?.chunk_documents(text)
    }

    // Aliases dos métodos originais.
    pub fn chunks_character(&self, text: &str) -> Result<Vec<String>> {
        self.chunk_with(text, Some("character"))
    }
    pub fn chunks_recursive(&self, text: &str) -> Result<Vec<String>> {
        self.chunk_with(text, Some("recursive"))
    }
    pub fn chunks_token(&self, text: &str) -> Result<Vec<String>> {
        self.chunk_with(text, Some("token"))
    }
    pub fn chunks_semantic(&self, text: &str) -> Result<Vec<String>> {
        self.chunk_with(text, Some("semantic"))
    }
}

impl Chunker for ChunkManager {
    fn name(&self) -> &str {
        &self.config.strategy
    }
    fn chunk_size(&self) -> usize {
        self.config.chunk_size
    }
    fn chunk_overlap(&self) -> usize {
        self.config.chunk_overlap
    }
    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        self.chunk_with(text, None)
    }
    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        self.chunk_documents_with(text, None)
    }
}
