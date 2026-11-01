# Testes do crate `embedding`

Situação em 2026-10-31. Referência Python usada na paridade: chatbot_rag `src/embedding`, com
torch 2.12, transformers 5.8.1, sentence-transformers 5.5.0 e scikit-learn 1.8.

Nenhum teste do crate baixa modelos: os testes com modelo (`tests/models.rs` e a paridade ONNX ×
PyTorch) foram removidos depois de rodados uma vez. O único teste com modelo que sobrou é o da API
(`api/tests/embedding.rs`), que usa o MiniLM do cache local e é pulado quando ele não existe.

## Já testado

### Sem modelo (rodam no CI: `cargo test -p embedding`)

| Teste | O que cobre | Resultado |
|---|---|---|
| Unitários (`--lib`, 21) | config e apelidos, recusa de provedores pagos, analisador do sklearn, MurmurHash3, modos de pooling, `_finalize`, SVD/Jacobi, leitor de `pytorch_model.bin` (recusa `os.system`), fp16, pré-processamento CLIP/SigLIP, host do Ollama | ok |
| `tests/strategies.rs` (8) | contrato completo nas 19 estratégias (embedder injetado), `_finalize` (corte, normalização, NaN, linhas a menos), erros de configuração, TF-IDF/hashing, BM25 (fórmula à mão, ranking, esparso), LSA (dimensões, determinismo) | ok |
| `tests/strategies.rs` `async_contract` | `aembed`, `aembed_documents`, `aembed_query`, `aembed_images` | ok (só com `--all-features`) |
| `tests/parity.rs` sem modelo | 7 configurações de `tfidf` e 6 de `hashing` contra o sklearn (n-gramas de palavra/caractere, stop words, acentos, `min_df`/`max_df`, `binary`, normas, `alternate_sign`, colisões) | idêntico (diferença ≤ 1e-6) |
| `cargo clippy -p embedding --all-targets -- -D warnings` | features padrão, `--no-default-features` e `--all-features` | 0 avisos |

### Com modelo (rodados uma vez, em 2026-10-31; testes removidos)

**Paridade ONNX × PyTorch**, 13 casos, todos ok:

| Estratégia | Modelo | Cosseno mínimo exigido |
|---|---|---|
| `sentence_transformer` | all-MiniLM-L6-v2 | 0,9999 |
| `sbert` (`dimensions: 128`, `normalize`) | all-MiniLM-L6-v2 | 0,9999 |
| `e5` (prefixos `query:`/`passage:`) | all-MiniLM-L6-v2 | 0,9999 |
| `nomic` (prefixos, `query_prefix` customizado) | all-MiniLM-L6-v2 | 0,9999 |
| `multilingual` | paraphrase-multilingual-MiniLM-L12-v2 | 0,9999 |
| `huggingface` (`mean`, `max`, `cls`) | all-MiniLM-L6-v2 | 0,9999 |
| `bge` (pooling CLS + Normalize) | BAAI/bge-small-en-v1.5 | 0,9999 |
| `instructor` (pares, `Dense` do `pytorch_model.bin`, espelho `Xenova`) | hkunlp/instructor-base | 0,9999 |
| `sparse` | prithivida/Splade_PP_en_v1 | 0,999 |
| `clip` (textos e imagens, espelho `Xenova`) | openai/clip-vit-base-patch32 | 0,995 |
| `siglip` (textos e imagens, espelho `Xenova`) | google/siglip-base-patch16-224 | 0,995 |

**Modelos padrão**, conferindo dimensão e busca do documento certo:

| Teste | Modelo | Também cobre | Resultado |
|---|---|---|---|
| `sentence_transformer` | all-MiniLM-L6-v2 (`sentence_transformer`, `sbert`, `huggingface`) | — | ok |
| `nomic` | nomic-embed-text-v1.5 | arquitetura própria em ONNX; `dimensions: 256` + `normalize` | ok |
| `gte` | gte-multilingual-base | ONNX do espelho `onnx-community`, configuração do original | ok |
| `jina` | jina-embeddings-v3 | entrada `task_id`, dados externos `.onnx_data`, vetor de consulta ≠ de documento | ok |

### Workspace e API (sem rede, 2026-10-31)

| Verificação | Resultado |
|---|---|
| `cargo deny --offline check` (padrão e `--all-features`) | ok (advisories, bans, licenses, sources) |
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 avisos |
| `cargo test --workspace --locked --offline` | 207 passaram, 0 falharam, 12 ignorados (os que baixavam modelos; removidos depois) |
| `api/tests/embedding.rs` (7) | catálogo e disponibilidade, `hashing`/`tfidf`/`bm25`/`lsa` pelo worker real, saída do `/v1/chunk` como entrada, validação e erros sem detalhes internos, 1 pedido por usuário, worker morto por sinal (422 + alerta), MiniLM do cache local pelo worker com sandbox (384 dimensões, busca certa) |
| `api/tests/docs.rs` | rotas novas documentadas e campos das respostas iguais ao OpenAPI |
| `kubectl kustomize` e `nginx -t` (imagem local) | ok |

## Falta testar

### Modelos padrão nunca rodados

| Modelo | Download | Por que falta pouco |
|---|---|---|
| `bge` (BAAI/bge-m3) | 2,2 GB | dados externos já verificados no `jina`; CLS verificado no bge-small |
| `e5` (multilingual-e5-base) | 1,1 GB | prefixos verificados com MiniLM |
| `instructor` (instructor-large) | 1,3 GB | mesmo caminho do instructor-base (espelho + `Dense`) |

### Caminhos de código sem nenhum teste

- **`ollama`:** nenhuma chamada real (precisa de um Ollama local com `nomic-embed-text`).
- **`device: "cuda"` / feature `cuda`:** nem compilado.
- **`Dense` lido de `model.safetensors`** (LaBSE, distiluse): só o caminho do `pytorch_model.bin`
  foi exercitado.
- **`model_name` como diretório local**, **modo offline** (só cache) e `model_options.revision`.
- **`onnx_file`** com exports quantizados ou fp16 (a saída fp16 é recusada com erro; não verificado).
- **Entradas raras do grafo:** padding à esquerda, `position_ids`, entradas `int32`.
- **Imagens:** só PNG por caminho; faltam JPEG/WebP/TIFF/GIF e `ImageInput::Bytes`.
- **Volume:** textos longos e lotes grandes (sublotes do SPLADE por orçamento de memória, ordenação
  por tamanho), tempo e memória.
- **Assíncrono com modelo real:** só testado com embedder injetado.
- **Mensagens de erro:** modelo inexistente, Hub inacessível, ONNX sem a saída esperada.

### Paridade não feita

- `jina`, `nomic`, `gte`, `bge-m3`, `e5-base`: só o teste de busca, sem comparar vetores com o
  PyTorch. No `jina` a comparação não faria sentido: o adaptador Python não escolhe tarefa
  (diferença intencional, ver README).
- `bm25`: só conferido com a fórmula à mão; falta comparar com uma implementação de referência
  (`rank_bm25`, Lucene).
- `lsa`: não comparável ao `TruncatedSVD` do sklearn (aleatório por padrão); falta avaliar a
  qualidade num acervo real.
- `tfidf` com empate de frequência no corte de `max_features`: diferença conhecida, sem teste.

### Integração

- Não rodados: CI (GitHub Actions) e build da imagem Docker com o crate.
- Endpoints com modelos grandes, CLIP/SigLIP (imagens em base64) e `EMBEDDING_DEVICE=cuda` não foram
  exercitados pela API; só o MiniLM local.
- O script que gerou `tests/fixtures/parity.json` ficou fora do repositório; sem ele, a fixture
  não pode ser regenerada.
