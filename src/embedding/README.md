# embedding

Estratégias de embeddings para RAG em Rust: o port do `src/embedding` do chatbot_rag (mesmos
nomes, apelidos, configuração e contrato) **sem os provedores pagos**, mais duas estratégias **só
de lógica** (`bm25`, `lsa`). É Rust puro em tempo de execução: os modelos rodam em ONNX Runtime,
em processo, no mesmo ambiente ONNX do OCR (docling) e do chunking.

Cada estratégia fica num arquivo próprio em `src/strategies/`. O `EmbeddingManager` escolhe a
estratégia pela configuração (o mesmo formato do `embedding:` do YAML do chatbot) e carrega o
modelo no primeiro uso.

```rust
use embedding::{Embedder, EmbeddingManager, InputType};
use serde_json::json;

let m = EmbeddingManager::from_value(json!({
    "strategy": "bge", "model_name": "BAAI/bge-m3", "normalize": true,
}))?;
let docs: Vec<Vec<f32>> = m.embed_documents(&["documento A", "documento B"])?;
let query: Vec<f32> = m.embed_query("minha consulta")?;
let sparse = m.embed_sparse(&["texto"], InputType::Document)?; // índices + valores
```

Todo embedder implementa o contrato `Embedder`: `embed` (documento ou consulta), `embed_docs`,
`embed_documents`, `embed_query`, `embed_images` (multimodais; erro de capacidade nos outros),
`embed_sparse` e, com a feature `async`, `aembed`, `aembed_documents`, `aembed_query` e
`aembed_images`. O manager aplica o `_finalize` do Python: uma linha por texto, só valores
finitos, corte em `dimensions` (Matryoshka) e normalização opcional.

## Estratégias

| Estratégia | Arquivo | O que faz | Modelo padrão |
|---|---|---|---|
| `sentence_transformer` | `sentence_transformer.rs` | Qualquer modelo do sentence-transformers: pipeline do `modules.json` (Transformer → Pooling → Dense → Normalize) e prompts de consulta/documento | `all-MiniLM-L6-v2` |
| `sbert` | `sbert.rs` | SBERT clássicos (MiniLM, MPNet...) | `all-MiniLM-L6-v2` |
| `bge` | `bge.rs` | BGE da BAAI (CLS + normalização) | `BAAI/bge-m3` |
| `gte` | `gte.rs` | GTE da Alibaba (ONNX do espelho `onnx-community`) | `Alibaba-NLP/gte-multilingual-base` |
| `e5` | `e5.rs` | E5, com `query: ` / `passage: ` | `intfloat/multilingual-e5-base` |
| `multilingual` | `multilingual.rs` | Multilíngues do sentence-transformers (LaBSE e distiluse com `Dense`) | `paraphrase-multilingual-MiniLM-L12-v2` |
| `instructor` | `instructor.rs` | INSTRUCTOR: par `[instrução, texto]`, `Dense` lido do `pytorch_model.bin` | `hkunlp/instructor-large` |
| `jina` | `jina.rs` | Jina v3: adaptador LoRA por tarefa (`task_id`) e prompt da tarefa | `jinaai/jina-embeddings-v3` |
| `nomic` | `nomic.rs` | Nomic Embed, com `search_query: ` / `search_document: ` | `nomic-ai/nomic-embed-text-v1.5` |
| `huggingface` | `huggingface.rs` | Encoder qualquer com pooling `mean`, `max` ou `cls` | `all-MiniLM-L6-v2` |
| `sparse` | `sparse.rs` | SPLADE: `log(1 + relu(logits))` máximo por termo do vocabulário | `prithivida/Splade_PP_en_v1` |
| `tfidf` | `tfidf.rs` | `TfidfVectorizer` do scikit-learn | — |
| `hashing` | `hashing.rs` | `HashingVectorizer` do scikit-learn (MurmurHash3) | — |
| `clip` | `clip.rs` | Texto e imagem no mesmo espaço (ONNX do espelho `Xenova`) | `openai/clip-vit-base-patch32` |
| `siglip` | `siglip.rs` | Como o CLIP, perda sigmoide (ONNX do espelho `Xenova`) | `google/siglip-base-patch16-224` |
| `ollama` | `ollama.rs` | Servidor Ollama local (`/api/embed`) | `nomic-embed-text` |
| `custom` | `custom.rs` | Embedder injetado (`with_embedder` ou `from_fn`) | — |
| **`bm25`** | `bm25.rs` | Okapi BM25 como vetores: consulta·documento = nota BM25 (busca híbrida) | — |
| **`lsa`** | `lsa.rs` | TF-IDF + SVD truncado: vetores densos sem modelo | — |

As opções de cada estratégia estão no topo do arquivo dela. Opção desconhecida é recusada,
como o `TypeError` de argumento inesperado no Python.


**Novas, só de lógica:** `bm25` é a perna lexical de uma busca híbrida (vetores esparsos cujo
produto escalar é a nota BM25); `lsa` dá vetores densos que aproximam termos que aparecem nos
mesmos contextos do acervo, sem modelo nem rede. As duas, como o `tfidf`, aprendem o vocabulário
no primeiro lote de documentos, e consultar antes disso é erro.

## Features

| Feature | Padrão | Para |
|---|---|---|
| `onnx` | sim | famílias densas, `huggingface` e `sparse` (ONNX Runtime + tokenizadores + Hugging Face Hub) |
| `vision` | sim | `clip`, `siglip` (decodificação e pré-processamento de imagem) |
| `ollama` | não | `ollama` (cliente HTTP) |
| `async` | não | contrato assíncrono sobre tokio |
| `cuda` | não | `device: "cuda"` (ONNX Runtime com CUDA) |

Sem features, ficam `tfidf`, `hashing`, `bm25`, `lsa` e `custom`, sem nenhuma dependência nativa.


## Testes

```bash
cargo test -p embedding                     # unitários, contrato, paridade sem modelo (sem rede)
cargo test -p embedding --all-features      # + contrato assíncrono e cliente Ollama
```
