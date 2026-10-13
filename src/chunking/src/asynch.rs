//! Contrato assíncrono (`achunk`, `abatch`, `atransform_documents`): o trabalho roda numa thread
//! de bloqueio do tokio, como o `asyncio.to_thread` do Python.

use std::sync::Arc;

use crate::chunker::Chunker;
use crate::document::Document;
use crate::error::{ChunkError, Result};

fn join_err(e: tokio::task::JoinError) -> ChunkError {
    ChunkError::Model(format!("tarefa de chunking: {e}"))
}

pub async fn achunk(chunker: Arc<dyn Chunker>, text: String) -> Result<Vec<String>> {
    tokio::task::spawn_blocking(move || chunker.chunk(&text)).await.map_err(join_err)?
}

/// Vários textos em paralelo, preservando a ordem.
pub async fn abatch(chunker: Arc<dyn Chunker>, texts: Vec<String>) -> Result<Vec<Vec<String>>> {
    let handles: Vec<_> = texts
        .into_iter()
        .map(|t| {
            let c = chunker.clone();
            tokio::task::spawn_blocking(move || c.chunk(&t))
        })
        .collect();
    let mut out = Vec::with_capacity(handles.len());
    for h in handles {
        out.push(h.await.map_err(join_err)??);
    }
    Ok(out)
}

pub async fn atransform_documents(chunker: Arc<dyn Chunker>, documents: Vec<Document>) -> Result<Vec<Document>> {
    tokio::task::spawn_blocking(move || chunker.transform_documents(&documents)).await.map_err(join_err)?
}
