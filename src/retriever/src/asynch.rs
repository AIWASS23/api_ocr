//! Contrato assíncrono (`ainvoke`, `abatch`, `astream`): a busca roda numa thread de bloqueio do
//! tokio, como o `asyncio.to_thread` do Python; `abatch` roda as consultas em paralelo.

use std::sync::Arc;

use chunking::Document;
use tokio::sync::mpsc;

use crate::error::{RetrievalError, Result};
use crate::retriever::Retriever;

fn join_err(e: tokio::task::JoinError) -> RetrievalError {
    RetrievalError::model(format!("retrieval task: {e}"))
}

pub async fn ainvoke(retriever: Arc<dyn Retriever>, query: String) -> Result<Vec<Document>> {
    tokio::task::spawn_blocking(move || retriever.invoke(&query)).await.map_err(join_err)?
}

/// Várias consultas em paralelo, com o resultado (ou erro) de cada uma na ordem da entrada
/// (`return_exceptions=True`).
pub async fn abatch_results(retriever: Arc<dyn Retriever>, queries: Vec<String>) -> Vec<Result<Vec<Document>>> {
    let tasks: Vec<_> = queries.into_iter().map(|q| tokio::spawn(ainvoke(retriever.clone(), q))).collect();
    let mut out = Vec::with_capacity(tasks.len());
    for t in tasks {
        out.push(t.await.map_err(join_err).and_then(|r| r));
    }
    out
}

/// Várias consultas em paralelo; o primeiro erro (na ordem da entrada) vira o resultado.
pub async fn abatch(retriever: Arc<dyn Retriever>, queries: Vec<String>) -> Result<Vec<Vec<Document>>> {
    abatch_results(retriever, queries).await.into_iter().collect()
}

/// O resultado por um canal (um único item, como no Python).
pub fn astream(retriever: Arc<dyn Retriever>, query: String) -> mpsc::Receiver<Result<Vec<Document>>> {
    let (tx, rx) = mpsc::channel(1);
    tokio::spawn(async move {
        let _ = tx.send(ainvoke(retriever, query).await).await;
    });
    rx
}
