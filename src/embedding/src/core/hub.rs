//! Arquivos de um modelo: diretório local ou repositório do Hugging Face Hub (cache em
//! `HF_HOME`, o mesmo do Python). Quando o repositório não publica ONNX (`openai/clip-vit-*`,
//! `google/siglip-*`, `hkunlp/instructor-*`, `Alibaba-NLP/gte-multilingual-base`), o ONNX vem de
//! um espelho com o mesmo nome (`onnx-community/<nome>`, `Xenova/<nome>`) e a configuração
//! (pooling, Dense, prompts) continua vindo do repositório original.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Cache, Repo, RepoType};
use serde_json::Value;

use crate::error::{EmbeddingError, Result};

/// Donos de espelhos ONNX consultados, em ordem.
const MIRROR_OWNERS: &[&str] = &["onnx-community", "Xenova"];

struct HubRepo {
    id: String,
    api: ApiRepo,
    cache: hf_hub::CacheRepo,
    /// Arquivos do repositório (`None`: sem rede; vale o que estiver no cache).
    files: Option<HashSet<String>>,
}

impl HubRepo {
    fn open(api: &Api, cache: &Cache, id: &str, revision: Option<&str>) -> Self {
        let repo = match revision {
            Some(r) => Repo::with_revision(id.to_string(), RepoType::Model, r.to_string()),
            None => Repo::new(id.to_string(), RepoType::Model),
        };
        let api_repo = api.repo(repo.clone());
        let files = api_repo.info().ok().map(|i| i.siblings.into_iter().map(|s| s.rfilename).collect());
        Self { id: id.to_string(), api: api_repo, cache: cache.repo(repo), files }
    }

    fn exists(&self) -> bool {
        self.files.is_some()
    }

    fn has(&self, file: &str) -> bool {
        match &self.files {
            Some(f) => f.contains(file),
            None => self.cache.get(file).is_some(),
        }
    }

    fn get(&self, file: &str) -> Result<Option<PathBuf>> {
        if !self.has(file) {
            return Ok(None);
        }
        if let Some(p) = self.cache.get(file) {
            return Ok(Some(p));
        }
        self.api.get(file).map(Some).map_err(|e| EmbeddingError::model(format!("'{}/{file}': {e}", self.id)))
    }

    /// Arquivos de dados externos de um ONNX (`model.onnx_data`, `model.onnx.data`...).
    fn external_data(&self, onnx: &str) -> Vec<String> {
        self.files
            .iter()
            .flatten()
            .filter(|f| f.as_str() != onnx && f.starts_with(onnx) && (f[onnx.len()..].starts_with("_data") || f[onnx.len()..].starts_with(".data")))
            .cloned()
            .collect()
    }
}

struct HubSource {
    primary: HubRepo,
    api: Api,
    cache: Cache,
    mirrors: OnceLock<Vec<HubRepo>>,
}

enum Source {
    Local(PathBuf),
    Hub(Box<HubSource>),
}

pub struct ModelFiles {
    name: String,
    source: Source,
}

impl ModelFiles {
    /// `name_or_path`: diretório local ou id do Hub (`dono/modelo`). `revision`: branch, tag ou
    /// commit no Hub.
    pub fn open(name_or_path: &str, revision: Option<&str>) -> Result<Self> {
        let path = Path::new(name_or_path);
        if path.is_dir() {
            return Ok(Self { name: name_or_path.to_string(), source: Source::Local(path.to_path_buf()) });
        }
        if path.is_file() {
            return Err(EmbeddingError::config(format!("model_name '{name_or_path}' must be a model directory or a Hub id, not a file")));
        }
        let cache = Cache::from_env();
        let api = ApiBuilder::from_cache(cache.clone())
            .with_progress(false)
            .build()
            .map_err(|e| EmbeddingError::model(format!("Hugging Face Hub: {e}")))?;
        let primary = HubRepo::open(&api, &cache, name_or_path, revision);
        if !primary.exists() && primary.cache.get("config.json").is_none() && primary.cache.get("tokenizer.json").is_none() {
            return Err(EmbeddingError::model(format!(
                "model '{name_or_path}' not found: not a local directory and not reachable on the Hugging Face Hub (offline and not cached)"
            )));
        }
        Ok(Self { name: name_or_path.to_string(), source: Source::Hub(Box::new(HubSource { primary, api, cache, mirrors: OnceLock::new() })) })
    }

    /// Espelhos ONNX, abertos só quando o original não tem o arquivo pedido.
    fn mirrors(&self) -> &[HubRepo] {
        match &self.source {
            Source::Local(_) => &[],
            Source::Hub(h) => h.mirrors.get_or_init(|| {
                let base = self.name.rsplit('/').next().unwrap_or(&self.name);
                MIRROR_OWNERS
                    .iter()
                    .map(|owner| format!("{owner}/{base}"))
                    .filter(|id| *id != self.name)
                    .map(|id| HubRepo::open(&h.api, &h.cache, &id, None))
                    .collect()
            }),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Arquivo do repositório original (configuração do sentence-transformers etc.).
    pub fn primary(&self, file: &str) -> Result<Option<PathBuf>> {
        match &self.source {
            Source::Local(dir) => Ok(Some(dir.join(file)).filter(|p| p.is_file())),
            Source::Hub(h) => h.primary.get(file),
        }
    }

    /// Arquivo do original ou, se faltar, de um espelho.
    pub fn any(&self, file: &str) -> Result<Option<PathBuf>> {
        if let Some(p) = self.primary(file)? {
            return Ok(Some(p));
        }
        for m in self.mirrors() {
            if let Some(p) = m.get(file)? {
                return Ok(Some(p));
            }
        }
        Ok(None)
    }

    pub fn require(&self, file: &str) -> Result<PathBuf> {
        self.any(file)?.ok_or_else(|| EmbeddingError::model(format!("model '{}' has no '{file}'", self.name)))
    }

    /// JSON do repositório original (ou do espelho, se `any`).
    pub fn json(&self, file: &str, any: bool) -> Result<Option<Value>> {
        let path = if any { self.any(file)? } else { self.primary(file)? };
        let Some(path) = path else { return Ok(None) };
        let text = std::fs::read_to_string(&path).map_err(|e| EmbeddingError::model(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&text).map(Some).map_err(|e| EmbeddingError::model(format!("{}: {e}", path.display())))
    }

    /// O primeiro ONNX entre `candidates` (no original, depois nos espelhos), com os dados
    /// externos baixados ao lado. `explicit`: caminho escolhido em `options.onnx_file`
    /// (relativo ao repositório ou absoluto).
    pub fn onnx(&self, candidates: &[&str], explicit: Option<&str>) -> Result<PathBuf> {
        if let Some(e) = explicit {
            let p = Path::new(e);
            if p.is_absolute() {
                return if p.is_file() { Ok(p.to_path_buf()) } else { Err(EmbeddingError::model(format!("onnx_file '{e}' not found"))) };
            }
        }
        let wanted: Vec<&str> = explicit.map_or_else(|| candidates.to_vec(), |e| vec![e]);
        match &self.source {
            Source::Local(dir) => wanted.iter().map(|f| dir.join(f)).find(|p| p.is_file()).ok_or_else(|| {
                EmbeddingError::model(format!("model directory '{}' has none of: {}", dir.display(), wanted.join(", ")))
            }),
            Source::Hub(h) => {
                for file in &wanted {
                    if let Some(path) = Self::fetch_onnx(&h.primary, file)? {
                        return Ok(path);
                    }
                }
                let mirrors = self.mirrors();
                for repo in mirrors {
                    for file in &wanted {
                        if let Some(path) = Self::fetch_onnx(repo, file)? {
                            return Ok(path);
                        }
                    }
                }
                let mut searched = vec![self.name.clone()];
                searched.extend(mirrors.iter().filter(|m| m.exists()).map(|m| m.id.clone()));
                Err(EmbeddingError::model(format!(
                    "no ONNX export of '{}' found ({}) in: {}. Export one (optimum-cli export onnx) and use a local directory or options.onnx_file",
                    self.name,
                    wanted.join(", "),
                    searched.join(", ")
                )))
            }
        }
    }

    fn fetch_onnx(repo: &HubRepo, file: &str) -> Result<Option<PathBuf>> {
        let Some(path) = repo.get(file)? else { return Ok(None) };
        for extra in repo.external_data(file) {
            repo.get(&extra)?;
        }
        Ok(Some(path))
    }
}
