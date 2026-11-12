# Testes do crate `retriever`

Situação em 2026-11-12. Referência Python usada na paridade: chatbot_rag `src/retrieval`, com
langchain-core 1.7.1, numpy 2.5.4 e scikit-learn (cache do uv, sem rede).

Nenhum teste baixa modelos nem acessa a rede. O embedder, o LLM, o gerador de consultas, o
reranker, o compressor, o roteador e os providers dos testes são funções determinísticas
(`tests/common/mod.rs`).

## Já testado

| Teste | O que cobre | Resultado |
|---|---|---|
| Unitários (`--lib`, 8) | configuração, apelidos e chaves legadas, validação, `tokens` e `difflib` como no Python, datas ISO, marcadores de lista do LLM, JSON na resposta do LLM | ok |
| `tests/parity.rs` (62 casos + 12 legados) | saída do `RetrievalManager` do Python, documento a documento: conteúdo, ordem e todos os metadados (notas com tolerância de float32). Cobre as 44 estratégias, exceto `multimodal`, além de filtros, limiar, MMR, parâmetros do BM25, fusões com pesos e `rrf_k`, consultas geradas por LLM e por gerador, base trocada (`base_strategy`), `parent_store`, vizinhos, compressão e reranking por LLM, recência (`decay_rate` 0 e, só a ordem, o padrão), providers, roteador, `fallback`, `cascade` e os 12 métodos legados | idêntico |
| `tests/strategies.rs` (17 + 1 assíncrono) | as 44 estratégias com tudo injetado e o contrato completo (`invoke`, `batch`, `batch_results`, `stream`, `load_retriever`...), catálogo (44 estratégias, 118 providers), erros de configuração e de dependência, limite de aninhamento, `fallback` com todos os motivos, parâmetros entregues ao provider (`k`, `fetch_k`, `search_type`, limiar, filtros, coleção, vetor), componentes injetados na fusão, alterações por chamada sem mutar o manager, recência com fusos e datas futuras, `parent_document` sobre a saída real do `parent_child` do chunking, self-query nativo com LLM, busca por vetor, `embedding` sem modelo (`hashing`) e notas do LLM (`7,5`, negativos, sem número) | ok |
| `tests/llm.rs` (3, feature `llm`) | cliente compatível com OpenAI contra um servidor HTTP local: rota, corpo (`model`, `temperature`, `max_tokens`, `extra`), sem `Authorization` quando não há chave configurada, `Bearer` com `api_key`, erro HTTP com a mensagem do servidor, resposta só com raciocínio, conteúdo vazio | ok |
| `api/tests/retrieval.rs` (6) | `GET /v1/retrieval/strategies` (disponibilidade, motivos, `requires`, embeddings aceitos) e `POST /v1/retrieval` pelo worker isolado: BM25 com várias consultas e filtros, densa e híbrida com `hashing`, RRF lexical, `fallback` e `adaptive` sem embedding, recência, cascata, `generated_queries` (`multi_query`, `hyde`), saída do `/v1/chunk` (`parent_child`) no `parent_document`, 28 pedidos inválidos (400/413/422, sem detalhes internos), 1 pedido por usuário e worker morto por sinal (422 + alerta) | ok |
| `cargo clippy -p retriever --all-targets -- -D warnings` | features padrão, `--no-default-features`, `llm`, `async` e `--all-features` | 0 avisos |

A fixture `tests/fixtures/parity.json` é gerada por `tests/fixtures/gen_parity.py` (o comando
está no topo do script). O teste foi conferido com a fixture alterada de propósito (nota, texto
e ordem): ele acusa a diferença.

## Falta testar

- **Cross-encoder real** (`rerank`/`cross_encoder_rerank` sem reranker injetado): nunca rodou.
  Exige baixar o `BAAI/bge-reranker-base` (ou um ONNX dele). A comparação com o
  `HuggingFaceCrossEncoder` também falta.
- **`multimodal` com CLIP/SigLIP** e imagens no acervo: só o caminho por texto e o erro sem
  embedder multimodal foram exercitados. Não há paridade: no Python, o `multimodal` depende do
  `db.search` do projeto.
- **Embedders com modelo** (`e5`, `bge`...) nas estratégias densas: só o embedder falso e o
  `hashing`.
- **LLM real** (Ollama, vLLM): só o servidor local de teste. Prompts reais podem trazer
  formatos de resposta que os testes não cobrem.
- **Self-query nativo × `SelfQueryRetriever`**: implementações diferentes, sem paridade (o do
  Python exige um vector store do LangChain).
- **Volume**: acervos grandes (a busca local calcula o cosseno contra todo o acervo a cada
  consulta, com cache de vetores) e o tempo do `fuzzy` em textos longos.
- **API com modelos**: `/v1/retrieval` só foi exercitada com embeddings sem modelo (`hashing`) e
  sem cross-encoder (`RETRIEVAL_RERANK_MODEL`).
