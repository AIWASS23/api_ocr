//! `multimodal` — busca de imagens e textos por texto, com um embedder multimodal (CLIP,
//! SigLIP). No acervo local, documento com imagem (`metadata.image`, `image_path` ou
//! `file_path`) entra pelo embedding da imagem; os demais pelo do texto. Sem acervo, busca no
//! provider (o Python passava o vetor da consulta ao `db.search`; aqui também há
//! `RetrievalManager::retrieve_by_vector`).

use chunking::Document;
use embedding::{ImageInput, InputType};

use crate::context::Ctx;
use crate::core::documents::with_score;
use crate::core::vectors::{argsort_desc, cosine_scores};
use crate::error::{RetrievalError, Result};

/// Caminho da imagem de um documento, se houver.
pub fn image_of(doc: &Document) -> Option<&str> {
    ["image", "image_path", "file_path"].iter().find_map(|k| doc.metadata.get(*k).and_then(|v| v.as_str()))
}

/// Notas de um vetor de consulta contra o acervo (imagens e textos).
pub fn score_corpus(ctx: &Ctx, query_vector: &[f32]) -> Result<Vec<Document>> {
    let embedder = ctx.embedder()?.ok_or_else(|| RetrievalError::missing("multimodal strategy requires a multimodal embedder (clip/siglip) or a provider"))?;
    let docs = ctx.corpus()?;
    let mut matrix: Vec<Vec<f32>> = vec![Vec::new(); docs.len()];
    let images: Vec<usize> = (0..docs.len()).filter(|&i| image_of(&docs[i]).is_some()).collect();
    if !images.is_empty() {
        if !embedder.supports_images() {
            return Err(RetrievalError::config("documents with images require a multimodal embedder (clip/siglip)"));
        }
        let inputs: Vec<ImageInput> = images.iter().map(|&i| ImageInput::from(image_of(&docs[i]).unwrap_or_default())).collect();
        for (&i, v) in images.iter().zip(embedder.embed_images(&inputs)?) {
            matrix[i] = v;
        }
    }
    let texts: Vec<usize> = (0..docs.len()).filter(|i| !images.contains(i)).collect();
    if !texts.is_empty() {
        let refs: Vec<&str> = texts.iter().map(|&i| docs[i].page_content.as_str()).collect();
        for (&i, v) in texts.iter().zip(ctx.embed(&embedder, &refs, InputType::Document)?) {
            matrix[i] = v;
        }
    }
    let scores = cosine_scores(&matrix, query_vector);
    Ok(argsort_desc(&scores).into_iter().take(ctx.cfg.k).map(|i| with_score(&docs[i], scores[i] as f64)).collect())
}

pub fn run(ctx: &Ctx, query: &str) -> Result<Vec<Document>> {
    if !ctx.cfg.documents.is_empty()
        && let Some(embedder) = ctx.embedder()?
    {
        if query.trim().is_empty() {
            return Err(RetrievalError::Input("query must be a non-empty string".into()));
        }
        let q = ctx.embed(&embedder, &[query], InputType::Query)?.remove(0);
        return score_corpus(ctx, &q);
    }
    crate::strategies::graph::provider_results(ctx, query)
}
