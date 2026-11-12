# retriever

Estratégias de recuperação (retrieval) para RAG em Rust: o port do `src/retrieval` do
chatbot_rag, com os mesmos nomes, apelidos, configuração e contrato. Os documentos são os
`chunking::Document` (o que o crate `chunking` produz). As estratégias densas usam o crate
`embedding`.

Cada estratégia fica num arquivo próprio em `src/strategies/`. O `RetrievalManager` escolhe a
estratégia pela configuração. Os parâmetros podem mudar a cada chamada.

```rust
use std::sync::Arc;
use retriever::{RetrievalManager, Retriever};
use serde_json::json;

let manager = RetrievalManager::from_value(json!({
    "strategy": "hybrid",
    "k": 5,
    "embedding": {"strategy": "e5"},          // ou .with_embedder(...)
    "documents": chunks,                        // saída do chunking (JSON ou Document)
}))?
.with_llm(Arc::new(|prompt: &str| meu_llm(prompt)));

let docs = manager.retrieve("praias de Fortaleza")?;
let lexical = manager.retrieve_with("CPF 123.456", json!({"strategy": "bm25", "k": 3}))?;
```

Cada resultado traz `metadata.retrieval_score` (quando a estratégia dá nota),
`retrieval_rank` (1, 2...) e `retrieval_strategy`. O contrato `Retriever` tem `retrieve`,
`load_retriever`, `invoke`, `batch`, `batch_results` (o `return_exceptions=True` do Python),
`stream` e `get_relevant_documents`. Com a feature `async`, `asynch` tem `ainvoke`, `abatch`,
`abatch_results` e `astream`. Os 12 métodos nomeados do manager original (`dense_retrieval`,
`retrieve_img`, `tf_idf_retrieval`, `bm25_retrieval`, `hybrid_retrieval`, `rerank_retrieval`,
`contextual_compression_retrieval`, `multi_query_retrieval`, `ensemble_retrieval`,
`self_query_retrieval`, `rephrase_query_retrieval`, `reprase_query_retrieval`) continuam
disponíveis.

## Estratégias

O que cada estratégia precisa: **E** = embedder (ou provider), **L** = LLM (ou o gerador
injetado indicado), **P** = provider injetado. Sem marca, roda só com os documentos.

| Família | Estratégia | O que faz | Precisa |
|---|---|---|---|
| Semântica | `dense` | Cosseno consulta × documento; `score_threshold` opcional | E |
| | `dense_mmr` | MMR: relevância × diversidade (`lambda_mult`, 0,5) | E |
| | `similarity_threshold` | Só notas ≥ `score_threshold` | E |
| | `multimodal` | Texto busca imagens (`metadata.image`/`image_path`/`file_path`) com CLIP/SigLIP | E multimodal |
| Lexical | `sparse` | BM25 no acervo local; sem acervo, o provider | — |
| | `bm25` | Okapi BM25 (`k1` 1,5, `b` 0,75) | — |
| | `bm25_plus` | BM25+ (`delta` 1) | — |
| | `tfidf` | TF-IDF do scikit-learn, unigramas e bigramas | — |
| | `keyword` | Contagem dos termos no texto | — |
| | `boolean` | Todos os termos (`operator: "and"`) ou algum (`"or"`) | — |
| | `fuzzy` | `difflib.SequenceMatcher` (tolera erros de digitação) | — |
| | `metadata` | Só `filters` | — |
| Fusão | `hybrid` | RRF dos componentes (padrão `dense` + `bm25`) | E |
| | `weighted_hybrid`, `rrf` | RRF com `weights` | E* |
| | `ensemble` | `EnsembleRetriever`: estratégias e providers injetados | E* |
| Consulta | `multi_query` | Variações da consulta, fundidas por RRF | L ou `QueryGenerator` |
| | `query_expansion` | Sinônimos e termos relacionados | L ou `QueryGenerator` |
| | `decomposition` | Subperguntas | L ou `QueryGenerator` |
| | `rephrase_query` | Reescrita para busca | L ou `QueryGenerator` |
| | `hyde` | Documento hipotético | L ou `QueryGenerator` |
| | `step_back` | Pergunta mais geral | L ou `QueryGenerator` |
| | `self_query` | Consulta → texto + filtro de metadados | construtor, ou L + `metadata_fields` |
| Hierárquica | `parent_document` | Acha o trecho, devolve o pai | `parent_store` ou `parent_content` |
| | `multi_vector` | Várias representações → documento original | idem |
| | `small_to_big` | Pai + vizinhos do trecho | idem + `with_neighbor_loader` |
| Reranking | `contextual_compression` | Reduz cada candidato ao trecho relevante | L ou `Compressor` |
| | `rerank`, `cross_encoder_rerank` | Cross-encoder `rerank_model` (ONNX) sobre `fetch_k` candidatos | modelo ou `Reranker` |
| | `colbert_rerank` | ColBERT | `Reranker` |
| | `llm_rerank` | Nota de 0 a 10 pelo LLM | L ou `Reranker` |
| | `diversity` | MMR | E |
| Tempo | `recency` | `exp(−decay_rate·dias)` pela data em `time_key` (`timestamp`) | — |
| | `time_weighted` | Relevância + recência | E* |
| Fontes | `graph`, `knowledge_graph`, `sql`, `web_search`, `knowledge_base` | Resultado do provider | P |
| Orquestração | `adaptive` | Códigos/números → `hybrid`; senão `dense`; sem embedder, `bm25` | — |
| | `router` | Roteador escolhe a estratégia | `with_router` |
| | `fallback` | Primeira estratégia que devolver algo (`strategies`) | — |
| | `cascade` | Estágios: busca → reranker / compressão / fusão (`stages`) | conforme os estágios |
| | `custom` | Retriever próprio | `with_custom` |

\* Depende dos componentes ou da estratégia base. Estratégias que refinam candidatos usam
`options.base_strategy` (padrão `dense`). As opções de cada estratégia estão no topo do arquivo
dela.

## Configuração

Chaves: `strategy` (`dense`), `provider` (`vector_db`), `k` (5), `fetch_k` (20), `top_n` (3),
`score_threshold`, `collection_name`, `documents`, `retrievers`, `weights`, `filters`
(igualdade, ou pertencer a uma lista), `rerank_model` (`BAAI/bge-reranker-base`),
`embedding` (configuração do crate `embedding`), `llm`, `options` e `provider_options`. Chaves
desconhecidas vão para `options`. As chaves legadas do Python continuam valendo:
`retrieval_strategy`, `retriever_provider`, `k_neighbors`, `k_neighbors_rerank`,
`top_n_rerank`, `embedding_manager`/`embedding_function`, `retrieval_funcs`/`retriever_funcs`.
Os apelidos também valem (`semantic`, `mmr`, `fusion`, `reprase`, `kg`...).

**Providers.** `local`, `vector_db`, `in_memory` (documentos + embedder) e `custom`. As outras
114 entradas do catálogo do Python são SDKs do LangChain (Chroma, Qdrant, pgvector, Pinecone,
Tavily...). Elas são reconhecidas pelo nome e recusadas com `ExternalProvider`: implemente
`SearchProvider` para a sua fonte e injete com `with_provider`.

**Injeção** (no Python, objetos em `options`/`provider_options`):

| Método | Para |
|---|---|
| `with_embedder` | Estratégias densas (no lugar de `embedding`) |
| `with_llm` | Reescrita, compressão, `llm_rerank`, self-query |
| `with_query_generator` | Consultas geradas sem LLM |
| `with_reranker`, `with_compressor` | Reranking e compressão |
| `with_provider` | Fonte de documentos (`SearchProvider`: `search` e, opcional, `documents`) |
| `with_component` | Provider extra nas fusões |
| `with_custom`, `with_router`, `with_query_constructor`, `with_neighbor_loader` | `custom`, `router`/`adaptive`, `self_query`, `small_to_big` |

Closures servem direto: `Arc::new(|prompt: &str| -> retriever::Result<String> { ... })`.

**LLM sem injeção** (feature `llm`): cliente compatível com OpenAI (Ollama, vLLM,
llama.cpp, LM Studio), `"llm": {"model": "qwen3", "base_url": "http://localhost:11434/v1",
"api_key_env": "MINHA_CHAVE", "max_tokens": 512, "extra": {...}}`. A chave só é enviada se
`api_key_env` (ou `api_key`) estiver na configuração.

## Diferenças em relação ao Python

- Data ISO sem fuso é UTC (no Python, hora local da máquina).
- `parent_document`/`multi_vector`/`small_to_big` sem `parent_store` usam
  `metadata.parent_content` (estratégia `parent_child` do chunking com
  `include_parent_content`).
- `self_query` sem construtor: com `metadata_fields` e LLM, o LLM devolve JSON
  `{"query", "filter"}` (no Python, o `SelfQueryRetriever` do LangChain, que exige um vector
  store).
- `multimodal` funciona no acervo local (CLIP/SigLIP). `retrieve_by_vector` busca por um vetor
  já calculado (o `retrieve_img` do Python).
- Alterações por chamada aceitam as chaves legadas (`retrieve_with(q, {"k_neighbors": 2})`). No
  Python, essas chaves caíam em `options`.
- Estratégias aninhadas têm limite de 16 níveis (`router` que escolhe `router`). No Python, o
  limite era um `RecursionError`.
- `colbert_rerank` não carrega o RAGatouille: injete o reranker.

## Features

| Feature | Padrão | Para |
|---|---|---|
| `onnx` | sim | Cross-encoder (`rerank`, `cross_encoder_rerank`) e modelos de embedding |
| `vision` | não | `multimodal` com CLIP/SigLIP |
| `llm` | não | Cliente de LLM compatível com OpenAI |
| `async` | não | `ainvoke`, `abatch`, `astream` sobre tokio |

Sem features, ficam todas as estratégias com dependências injetadas e as lexicais. As densas
também funcionam com `embedding` sem modelo (`hashing`, `tfidf`, `bm25`, `lsa`).

## Testes

```bash
cargo test -p retriever                     # unitários, contrato, paridade com o Python (sem rede)
cargo test -p retriever --all-features      # + contrato assíncrono e cliente de LLM (servidor local)
```

Detalhes em [TESTES.md](TESTES.md).
