//! `clip` — CLIP da OpenAI (`VisionLanguageAdapter`/`ClipEmbedder` do chatbot): textos e
//! imagens no mesmo espaço (512 dimensões no ViT-B/32), para busca de imagem por texto e
//! vice-versa. Texto: tokenizador BPE, até 77 tokens. Imagem: RGB, lado menor em 224 (bicúbico),
//! recorte central 224×224, escala e normalização do `preprocessor_config.json`.
//!
//! Os repositórios `openai/clip-*` não publicam ONNX: os grafos (`text_model.onnx` e
//! `vision_model.onnx`) vêm do espelho `Xenova/clip-*`.
//!
//! Padrão: `openai/clip-vit-base-patch32`. Diferença intencional: textos com mais de 77 tokens
//! são cortados (no Python, o modelo dá erro).

use crate::config::EmbeddingConfig;
use crate::core::hub::ModelFiles;
use crate::core::options::Options;
use crate::core::vision::DualEncoder;
use crate::embedder::{Embedder, ImageInput, InputType};
use crate::error::{EmbeddingError, Result};

pub struct ClipEmbedder {
    name: String,
    encoder: DualEncoder,
    batch_size: usize,
}

/// Carrega um modelo texto+imagem; `pad_to_max_length` é o que separa o SigLIP do CLIP.
pub fn load_dual(config: &EmbeddingConfig, strategy: &'static str, default_max_length: usize, pad_to_max_length: bool) -> Result<ClipEmbedder> {
    let mut o = Options::new(strategy, &config.options);
    let mut mo = Options::new("model_options", &o.object("model_options")?);
    let revision = mo.string("revision")?;
    mo.ignore(&["trust_remote_code", "token", "cache_dir", "local_files_only", "torch_dtype", "dtype"]);
    mo.finish()?;
    o.finish()?;
    let name = config.model_name.clone().ok_or_else(|| EmbeddingError::config(format!("model_name is required for strategy '{strategy}'")))?;
    let files = ModelFiles::open(&name, revision.as_deref())?;
    let encoder = DualEncoder::load(&files, &config.device, default_max_length, pad_to_max_length)?;
    Ok(ClipEmbedder { name, encoder, batch_size: config.batch_size })
}

pub fn load(config: &EmbeddingConfig) -> Result<ClipEmbedder> {
    load_dual(config, "clip", 77, false)
}

impl Embedder for ClipEmbedder {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn embed(&self, texts: &[&str], _input_type: InputType) -> Result<Vec<Vec<f32>>> {
        self.encoder.embed_texts(texts, self.batch_size)
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn embed_images(&self, images: &[ImageInput]) -> Result<Vec<Vec<f32>>> {
        self.encoder.embed_images(images, self.batch_size)
    }
}
