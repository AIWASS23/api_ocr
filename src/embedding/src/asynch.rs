//! Contrato assíncrono (`aembed`, `aembed_documents`, `aembed_query`, `aembed_images`): o trabalho
//! roda numa thread de bloqueio do tokio, como o `asyncio.to_thread` do Python.

use std::sync::Arc;

use crate::embedder::{Embedder, ImageInput, InputType};
use crate::error::{EmbeddingError, Result};

fn join_err(e: tokio::task::JoinError) -> EmbeddingError {
    EmbeddingError::model(format!("tarefa de embedding: {e}"))
}

pub async fn aembed(embedder: Arc<dyn Embedder>, texts: Vec<String>, input_type: InputType) -> Result<Vec<Vec<f32>>> {
    tokio::task::spawn_blocking(move || {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        embedder.embed(&refs, input_type)
    })
    .await
    .map_err(join_err)?
}

pub async fn aembed_documents(embedder: Arc<dyn Embedder>, documents: Vec<String>) -> Result<Vec<Vec<f32>>> {
    aembed(embedder, documents, InputType::Document).await
}

pub async fn aembed_query(embedder: Arc<dyn Embedder>, query: String) -> Result<Vec<f32>> {
    tokio::task::spawn_blocking(move || embedder.embed_query(&query)).await.map_err(join_err)?
}

pub async fn aembed_images(embedder: Arc<dyn Embedder>, images: Vec<ImageInput>) -> Result<Vec<Vec<f32>>> {
    tokio::task::spawn_blocking(move || embedder.embed_images(&images)).await.map_err(join_err)?
}
