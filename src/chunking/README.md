# chunking

Estratégias de chunking para RAG em Rust: o port das **25 estratégias** do `src/chunking` do
chatbot_rag (mesmos nomes, apelidos, configuração e resultados) e mais **8 estratégias
estruturais e contextuais**. É Rust puro: sem Python em tempo de execução nem bindings. Os
modelos (tokenizadores, embeddings, ONNX) rodam dentro do processo.

Cada estratégia fica num arquivo próprio em `src/strategies/`. O `ChunkManager` escolhe a
estratégia pela configuração, no mesmo formato do `indexing.chunking` do YAML do chatbot.

```rust
use chunking::{ChunkManager, Chunker};
use serde_json::json;

let m = ChunkManager::from_value(json!({
    "strategy": "token", "chunk_size": 256, "chunk_overlap": 64, "model_name": "BAAI/bge-m3",
}))?;
m.load_model(None)?;                                   // tokenizador (Hub ou caminho local)
let chunks: Vec<String> = m.chunk(&texto)?;
let docs = m.chunk_documents(&texto)?;                 // com metadados da estratégia
let outra = m.chunk_with(&texto, Some("sentence"))?;   // outra estratégia só nesta chamada
```

Todo chunker implementa o contrato `Chunker`: `chunk`, `split_text`, `create_documents`,
`split_documents`, `transform_documents`, `batch` e, com a feature `async`, `achunk`, `abatch`
e `atransform_documents`. A diferença em relação ao Python é `chunk_documents`, que devolve
cada chunk com os metadados da estratégia (títulos, artigo de lei, id do pai, embedding...);
nos métodos de documento, esses metadados são somados aos do documento de origem.

## Estratégias

| Estratégia | Arquivo | O que faz | Precisa de |
|---|---|---|---|
| `character` | `character.rs` | Tamanho fixo por caracteres (`separator`, regex opcional) | — |
| `recursive` | `recursive.rs` | `\n\n` → `\n` → espaço → caractere; com `language`, separadores de código | — |
| `word` / `sentence` / `paragraph` / `line` | um arquivo cada | Recursivo preferindo palavras / frases / parágrafos / linhas | — |
| `token` | `token.rs` | Recursivo medido em tokens Hugging Face | tokenizador |
| `token_tiktoken` | `token_tiktoken.rs` | Janelas de tokens tiktoken (`cl100k_base`...) | feature `tiktoken` |
| `token_sentence_transformer` | `token_sentence_transformer.rs` | Janelas no limite do modelo Sentence Transformers | tokenizador (carrega sozinho) |
| `semchunk` | `semchunk.rs` | Algoritmo do semchunk: separadores mais significativos primeiro | tokenizador |
| `semantic` | `semantic.rs` | `SemanticChunker`: corta onde o assunto muda (distância entre frases) | embeddings |
| `nltk` | `nltk.rs` | Frases pelo Punkt do NLTK (parâmetros do `punkt_tab`) | dados `punkt_tab` (opcional) |
| `spacy` | `spacy.rs` | Frases pelo sentencizer do spaCy (`pt`, `en`) | — |
| `markdown` | `markdown.rs` | Recursivo com separadores de Markdown | — |
| `markdown_header` | `markdown_header.rs` | Divide pelos títulos `#`…`######` | — |
| `markdown_syntax` | `markdown_syntax.rs` | Títulos, blocos de código e réguas horizontais | — |
| `html` | `html.rs` | HTML como código (corta antes de `<div`, `<p`...) | — |
| `html_header` / `html_section` / `html_semantic` | um arquivo cada | HTML por títulos / seções / preservando tabelas e listas | feature `html` |
| `json` | `json.rs` | `RecursiveJsonSplitter` | — |
| `latex` | `latex.rs` | Seções e ambientes do LaTeX | — |
| `code` / `python` / `javascript` | um arquivo cada | Código pela linguagem; JSX/Vue/Svelte por componente | — |
| **`markdown_header_contextual`** | `markdown_header_contextual.rs` | Títulos de Markdown com o caminho (`Doc > Cap. > Seção`) repetido em **cada** pedaço | — |
| **`docling_hybrid`** | `docling_hybrid.rs` | `HybridChunker` do docling.rs: item a item (lista e tabela inteiras), com títulos | feature `docling` + tokenizador |
| **`table`** | `table.rs` | Tabelas Markdown inteiras; grandes divididas repetindo o cabeçalho | — |
| **`parent_child`** | `parent_child.rs` | Filhos pequenos para a busca, apontando para o pai grande que vai ao LLM | — |
| **`contextual_retrieval`** | `contextual_retrieval.rs` | Um LLM escreve o contexto de cada chunk antes do texto | LLM (feature `llm`) |
| **`legal`** | `legal.rs` | Leis/decretos: um artigo por chunk com `Lei > Título > Capítulo > Seção` | — |
| **`qa`** | `qa.rs` | FAQ: um par pergunta/resposta por chunk | — |
| **`late_chunking`** | `late_chunking.rs` | Embedding de cada chunk calculado com o documento inteiro como contexto | embeddings por token (feature `onnx`) |

As opções de cada estratégia estão documentadas no topo do arquivo dela. Uma opção desconhecida
é recusada, como o `TypeError` de argumento inesperado no Python.

Apelidos: os mesmos do Python (`fixed_size`, `sliding_window`, `tiktoken`, `sentences`,
`markdown_headers`, `js`...) e, para as novas, `contextual_header`, `hybrid`, `tables`,
`small_to_big`, `contextual`, `lei`/`legislacao`, `faq` e `late`.

## Features

| Feature | Padrão | Para |
|---|---|---|
| `html` | sim | `html_header`, `html_section`, `html_semantic` (parser HTML5 do `scraper`) |
| `tiktoken` | sim | `token_tiktoken` (BPE embutido, funciona offline) |
| `hf-tokenizers` | sim | `token`, `token_sentence_transformer`, `semchunk`, `docling_hybrid` |
| `embeddings` | não | `semantic` com embeddings locais (`fastembed` + ONNX Runtime) |
| `onnx` | não | `late_chunking` com ONNX Runtime |
| `llm` | não | `contextual_retrieval` com servidor `/v1/chat/completions` (Ollama, vLLM, OpenAI...) |
| `docling` | não | `docling_hybrid` |
| `async` | não | contrato assíncrono sobre tokio |

Sem a feature necessária, a estratégia devolve `ChunkError::FeatureDisabled`; ela nunca falha em
silêncio.

## Modelos

- **Tokenizador** (`model_name`): id do Hugging Face Hub (baixado e guardado em `HF_HOME`) ou
  caminho local de um diretório com `tokenizer.json`. Use o **mesmo modelo do embedding**: no
  `config.yaml` atual do chatbot, o chunking mede com `multilingual-e5-large-instruct` e o
  embedding usa `BAAI/bge-m3`.
- **Embeddings** (`semantic`): `load_model("semantic")` usa o fastembed com o nome do modelo
  (`BAAI/bge-m3`, `intfloat/multilingual-e5-large`...). Também aceita espelhos ONNX
  (`Xenova/...`, `Qdrant/...`) e qualquer implementação de `Embedder` (`with_embedder`).
- **Embeddings por token** (`late_chunking`): `options.embedding_model` (ou `model_name`) com
  `onnx/model.onnx` e saída `last_hidden_state`; `options.max_tokens` (8192) é a janela.
- **LLM** (`contextual_retrieval`): `options.llm_model` (obrigatório), `llm_base_url`
  (`http://localhost:11434/v1`), `llm_api_key` ou `llm_api_key_env` (`OPENAI_API_KEY`),
  `llm_prompt`, `llm_max_tokens`, `llm_timeout_secs` e `llm_extra`, que vai para o corpo da
  requisição. Para modelos de raciocínio no Ollama (qwen3...) use
  `llm_extra: {"reasoning_effort": "none"}`: sem isso, o modelo gasta os tokens pensando e a
  resposta vem vazia, o que é tratado como erro.
- **NLTK** (`nltk`): os parâmetros do `punkt_tab` são procurados onde o NLTK procura
  (`NLTK_DATA`, `~/nltk_data`, `/usr/share/nltk_data`...) ou em `options.nltk_data`. Sem eles a
  estratégia usa `sentence`, como o chatbot faz.

## Paridade com o Python

`tests/fixtures/parity.json` tem **231 casos**: 25 estratégias, vários tamanhos, sobreposições e
opções, e entradas difíceis (português com abreviações, NBSP e `\r\n`, Markdown com tabela e
código, HTML com comentários e `font-size`, JSON com Unicode, LaTeX, Python, Go, JSX). Os casos
foram gerados rodando o **`ChunkManager` real do chatbot_rag**. O teste `tests/parity.rs` exige
que o Rust produza **exatamente** os mesmos chunks, e os 231 passam.

Diferenças intencionais:

- **`html_section`**: o Python deixa o marcador literal `#TITLE#` no início da primeira seção
  (um bug); aqui entra o `<title>` do documento (ou `options.title_placeholder`). O Python
  também passa o HTML pelo serializador do lxml, que insere `\n` entre elementos de bloco; aqui o
  texto sai sem esse ruído. Por isso o teste compara essa estratégia com os espaços colapsados.
- **`spacy`**: só `sentencizer` e só `pt`/`en` (modelos treinados do spaCy não rodam em Rust);
  outro idioma usa `sentence`. Não há o limite de 1.000.000 caracteres do spaCy.
- **`html_*`**: o parser é HTML5 (html5ever), não o `html.parser` do Python. Em HTML bem formado
  o resultado é o mesmo; em HTML quebrado a recuperação de erros pode diferir. Na allowlist do
  `html_semantic`, `html`, `head` e `body` nunca são removidos. `custom_handlers` (funções Python)
  não existe.
- **`load_model(None)`** carrega o modelo da estratégia configurada. O Python carrega tokenizador
  e embeddings juntos.
- Mapas dentro de `options` (`external_metadata`, `custom_header_patterns`) chegam com as chaves
  em ordem alfabética, porque o `serde_json` do workspace não preserva a ordem.
- **`code`**: `perl` dá erro, como no LangChain, que não tem separadores para Perl.

## Testes

```bash
cargo test -p chunking                                   # unitários, paridade, contrato e estratégias novas
cargo test -p chunking --all-features -- --include-ignored  # + modelos reais (baixa ~220 MB; Ollama p/ contextual)
```
