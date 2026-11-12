//! Contrato comum de um retriever — o `BaseRetrieval` do Python: `retrieve` e, prontos a partir
//! dele, `load_retriever`, `invoke`, `batch`, `stream` e `get_relevant_documents`. As versões
//! assíncronas (`ainvoke`, `abatch`, `astream`) ficam em `asynch` (feature `async`).

use chunking::Document;

use crate::error::Result;

/// Os métodos do contrato do Python (`RETRIEVER_CONTRACT`).
pub const RETRIEVER_CONTRACT: &[&str] =
    &["retrieve", "load_retriever", "invoke", "ainvoke", "batch", "abatch", "stream", "astream", "get_relevant_documents"];

/// Métodos nomeados do `RetrievalManager` original (`LEGACY_RETRIEVAL_METHODS`).
pub const LEGACY_RETRIEVAL_METHODS: &[&str] = &[
    "dense_retrieval",
    "retrieve_img",
    "tf_idf_retrieval",
    "bm25_retrieval",
    "hybrid_retrieval",
    "rerank_retrieval",
    "contextual_compression_retrieval",
    "multi_query_retrieval",
    "ensemble_retrieval",
    "self_query_retrieval",
    "rephrase_query_retrieval",
    "reprase_query_retrieval",
];

pub trait Retriever: Send + Sync {
    /// Os documentos mais relevantes para a consulta.
    fn retrieve(&self, query: &str) -> Result<Vec<Document>>;

    /// Carrega antecipadamente o que a configuração pede (modelos, clientes). Opcional: tudo
    /// também é carregado no primeiro uso.
    fn load_retriever(&self) -> Result<()> {
        Ok(())
    }

    /// Entrada única compatível com o LangChain.
    fn invoke(&self, query: &str) -> Result<Vec<Document>> {
        self.retrieve(query)
    }

    /// Várias consultas, na ordem; para no primeiro erro.
    fn batch(&self, queries: &[&str]) -> Result<Vec<Vec<Document>>> {
        queries.iter().map(|q| self.invoke(q)).collect()
    }

    /// Várias consultas com o erro de cada uma no lugar do resultado (`return_exceptions=True`).
    fn batch_results(&self, queries: &[&str]) -> Vec<Result<Vec<Document>>> {
        queries.iter().map(|q| self.invoke(q)).collect()
    }

    /// O resultado por uma interface de streaming (um único item, como no Python).
    fn stream(&self, query: &str) -> std::iter::Once<Result<Vec<Document>>> {
        std::iter::once(self.invoke(query))
    }

    /// Nome antigo do LangChain.
    fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        self.retrieve(query)
    }
}
