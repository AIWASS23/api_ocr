"""Gera tests/fixtures/parity.json rodando o RetrievalManager do chatbot_rag (Python).

Cada caso descreve a configuração em JSON e as dependências falsas que entram nela
("deps"); o teste Rust (tests/parity.rs) monta as mesmas dependências com o mesmo
comportamento e compara documentos, ordem e metadados (notas com tolerância).

Nenhum modelo é usado: o embedder conta palavras de um vocabulário fixo, e o LLM,
o gerador de consultas, o reranker etc. são funções determinísticas.

Uso (sem rede, com o cache do uv):

    CHATBOT_RAG=~/Área\\ de\\ trabalho/chatbot_rag \\
    uv run --no-project --offline --with langchain-core --with numpy --with scikit-learn \\
        python -I src/retriever/tests/fixtures/gen_parity.py src/retriever/tests/fixtures/parity.json
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.expanduser(os.environ["CHATBOT_RAG"]))

from langchain_core.documents import Document  # noqa: E402

from src.retrieval import RetrievalManager  # noqa: E402

VOCABULARY = (
    "python", "rust", "banana", "busca", "híbrida", "densa", "lexical",
    "fruta", "sintaxe", "linguagem", "tipagem", "praia",
)

DOCUMENTS = [
    {
        "page_content": "Python possui tipagem dinâmica e uma sintaxe simples.",
        "metadata": {"id": "python", "kind": "code", "parent_id": "guide", "timestamp": "2026-11-01T12:00:00+00:00"},
    },
    {
        "page_content": "Banana é uma fruta tropical amarela, doce e barata.",
        "metadata": {"id": "banana", "kind": "food", "timestamp": "2026-09-15T08:30:00Z"},
    },
    {
        "page_content": "Busca híbrida combina recuperação densa e lexical.",
        "metadata": {"id": "search", "kind": "search", "parent_id": "guide-search", "timestamp": 1793000000},
    },
    {
        "page_content": "Rust é uma linguagem com tipagem estática e sintaxe expressiva.",
        "metadata": {"id": "rust", "kind": "code", "parent_id": "guide", "timestamp": "2026-10-20T00:00:00-03:00"},
    },
    {
        "page_content": "A busca lexical BM25 pondera termos raros; a busca densa usa vetores.",
        "metadata": {"id": "bm25", "kind": "search", "parent_id": "guide-search"},
    },
    {
        "page_content": "Fortaleza tem praia, sol e fruta fresca o ano todo.",
        "metadata": {"id": "fortaleza", "kind": "travel", "timestamp": "2025-12-31T23:59:59+00:00"},
    },
    {
        "page_content": "Python e Rust: duas linguagem populares para busca e dados.",
        "metadata": {"id": "compare", "kind": "code", "parent_id": "guide", "timestamp": 0},
    },
]

PARENT_STORE = {
    "guide": {"page_content": "Guia completo de linguagens: Python e Rust.", "metadata": {"id": "guide"}},
    "guide-search": {"page_content": "Guia de busca: densa, lexical e híbrida.", "metadata": {"id": "guide-search"}},
}


# ---------------------------------------------------------------- dependências falsas
# O teste Rust reproduz cada uma destas funções.


class FakeEmbeddings:
    def embed(self, texts, input_type="document"):
        del input_type
        return [[float(t.lower().count(w)) for w in VOCABULARY] for t in texts]


class FakeLLM:
    def invoke(self, prompt):
        if prompt.startswith("Avalie a relevância"):
            doc = prompt.split("Documento: ", 1)[1].lower()
            hits = sum(doc.count(w) for w in ("python", "busca", "rust"))
            return f"Nota: {hits * 2},5 de 10"
        if prompt.startswith("Extraia somente"):
            doc = prompt.split("Documento: ", 1)[1]
            return "  " if "banana" in doc.lower() else "  " + " ".join(doc.split()[:3]) + "  "
        if "Modo: " in prompt:
            return "1. python sintaxe\n- busca densa\n* python\n\n2) busca densa\nfruta"
        if prompt.startswith("Reescreva"):
            return "  rust sintaxe  "
        if prompt.startswith("Escreva um documento"):
            return "Busca híbrida usa busca densa e lexical."
        if prompt.startswith("Gere uma pergunta mais geral"):
            return "linguagem de programação"
        raise AssertionError(f"prompt inesperado: {prompt!r}")


def query_generator(query, mode, count=None):
    if count is None:
        return {"rephrase": "rust tipagem", "hyde": "fruta amarela", "step_back": "busca"}[mode]
    return [f"{query} {mode}", "busca lexical", query, "fruta"][:count]


def reranker(query, documents, top_n):
    del query
    return list(reversed(documents))[:top_n]


def compressor(query, documents):
    del query
    return [d for d in documents if "a" in d.page_content][:1]


def query_constructor(query):
    return (query.replace("código", "").strip(), {"kind": "code"})


def neighbor_loader(document):
    return [Document(page_content=f"Vizinho de {document.metadata['id']}", metadata={"id": f"n-{document.metadata['id']}"})]


def router(query, strategies):
    return strategies[-1] if "fruta" in query else strategies[0]


class FakeRetriever:
    """Provider/custom: devolve o acervo em ordem inversa."""

    def invoke(self, query):
        del query
        return [Document(**d) for d in reversed(DOCUMENTS)]


DEPS = {
    "embedder": lambda c: c.__setitem__("embedding_function", FakeEmbeddings()),
    "llm": lambda c: c.__setitem__("llm", FakeLLM()),
    "query_generator": lambda c: c["options"].__setitem__("query_generator", query_generator),
    "reranker": lambda c: c["options"].__setitem__("reranker", reranker),
    "compressor": lambda c: c["options"].__setitem__("compressor", compressor),
    "query_constructor": lambda c: c["options"].__setitem__("query_constructor", query_constructor),
    "neighbor_loader": lambda c: c["options"].__setitem__("neighbor_loader", neighbor_loader),
    "router": lambda c: c["options"].__setitem__("router", router),
    "provider": lambda c: c.__setitem__("provider_options", {"retriever": FakeRetriever()}),
    "custom": lambda c: c["options"].__setitem__("executor", FakeRetriever()),
}

BASE = {"provider": "local", "documents": DOCUMENTS, "k": 3, "fetch_k": 5, "top_n": 2}
E = ["embedder"]


def case(name, strategy, query, deps=E, rank_only=False, **config):
    return {"name": name, "strategy": strategy, "query": query, "deps": list(deps), "rank_only": rank_only, "config": config}


CASES = [
    # Semântica
    case("dense", "dense", "python sintaxe"),
    case("dense_threshold", "dense", "python sintaxe", score_threshold=0.3),
    case("dense_filters", "dense", "busca python", filters={"kind": "code"}),
    case("dense_filter_list", "dense", "busca", filters={"kind": ["code", "travel"]}),
    case("dense_mmr", "dense_mmr", "busca densa lexical", options={"lambda_mult": 0.3}),
    case("similarity_threshold", "similarity_threshold", "rust", score_threshold=0.2),
    case("diversity", "diversity", "python rust busca", k=4),
    # Lexical
    case("sparse", "sparse", "busca densa"),
    case("bm25", "bm25", "busca híbrida densa"),
    case("bm25_params", "bm25", "busca python", options={"k1": 1.2, "b": 0.5}),
    case("bm25_plus", "bm25_plus", "busca lexical termos"),
    case("tfidf", "tfidf", "busca densa lexical"),
    case("tfidf_bigram", "tfidf", "tipagem estática", k=2),
    case("keyword", "keyword", "fruta praia"),
    case("boolean_and", "boolean", "python sintaxe"),
    case("boolean_or", "boolean", "banana rust", options={"operator": "or"}),
    case("fuzzy", "fuzzy", "fruta amarela"),
    case("metadata", "metadata", "ignorada", filters={"kind": "search"}),
    # Fusão
    case("hybrid", "hybrid", "python busca"),
    case("weighted_hybrid", "weighted_hybrid", "python busca", weights=[0.7, 0.3]),
    case("rrf", "rrf", "busca densa", weights=[1.0, 2.0, 0.5], options={"components": ["dense", "bm25", "tfidf"], "rrf_k": 10}),
    case("ensemble", "ensemble", "rust tipagem", retrievers=["bm25", "keyword"]),
    # Transformação de consulta
    case("multi_query_gen", "multi_query", "python", deps=E + ["query_generator"]),
    case("multi_query_llm", "multi_query", "python", deps=E + ["llm"]),
    case("query_expansion", "query_expansion", "busca", deps=E + ["llm"], options={"query_count": 2}),
    case("decomposition", "decomposition", "python e busca", deps=E + ["query_generator"], options={"base_strategy": "bm25"}),
    case("rephrase_gen", "rephrase_query", "python", deps=E + ["query_generator"]),
    case("rephrase_llm", "rephrase_query", "python", deps=E + ["llm"]),
    case("hyde", "hyde", "o que é busca híbrida?", deps=E + ["llm"]),
    case("step_back", "step_back", "rust é tipado?", deps=E + ["llm"], options={"base_strategy": "keyword"}),
    case("hyde_gen", "hyde", "qual fruta?", deps=E + ["query_generator"]),
    case("self_query_constructor", "self_query", "código python", deps=E + ["query_constructor"]),
    case("self_query_base", "self_query", "python", options={"base_strategy": "bm25"}),
    # Hierárquica
    case("parent_document", "parent_document", "python rust busca", options={"parent_store": PARENT_STORE}),
    case("parent_document_no_store", "parent_document", "fruta"),
    case("multi_vector", "multi_vector", "busca densa", options={"parent_store": PARENT_STORE, "base_strategy": "bm25"}),
    case("small_to_big", "small_to_big", "python", deps=E + ["neighbor_loader"], k=4, options={"parent_store": PARENT_STORE}),
    # Compressão e reranking
    case("compression", "contextual_compression", "busca", deps=E + ["compressor"]),
    case("compression_llm", "contextual_compression", "fruta python", deps=E + ["llm"], top_n=3, fetch_k=4),
    case("rerank", "rerank", "python", deps=E + ["reranker"]),
    case("cross_encoder_rerank", "cross_encoder_rerank", "busca", deps=E + ["reranker"], options={"base_strategy": "bm25"}),
    case("colbert_rerank", "colbert_rerank", "rust", deps=E + ["reranker"]),
    case("llm_rerank", "llm_rerank", "python busca", deps=E + ["llm"], top_n=3),
    case("llm_rerank_injected", "llm_rerank", "python", deps=E + ["llm", "reranker"]),
    # Tempo
    case("recency_flat", "recency", "qualquer", options={"decay_rate": 0}),
    case("recency", "recency", "qualquer", rank_only=True, k=7),
    case("time_weighted_flat", "time_weighted", "python", options={"decay_rate": 0}),
    # Fontes especializadas (provider injetado)
    *[case(s, s, "python", deps=["provider"], documents=[], filters={"kind": "code"}) for s in ("graph", "knowledge_graph", "sql", "web_search", "knowledge_base")],
    # Orquestração
    case("adaptive_codes", "adaptive", "python 3.12"),
    case("adaptive_dense", "adaptive", "python"),
    case("adaptive_lexical", "adaptive", "busca densa", deps=[]),
    case("adaptive_router", "adaptive", "fruta", deps=E + ["router"], options={"strategies": ["dense", "keyword"]}),
    case("router", "router", "python fruta", deps=E + ["router"], options={"strategies": ["dense", "fuzzy"]}),
    case("fallback", "fallback", "python", options={"strategies": ["custom", "boolean", "bm25"]}),
    case("cascade_default", "cascade", "python", deps=E + ["reranker"]),
    case("cascade_stages", "cascade", "busca densa", deps=E + ["llm"], options={"stages": ["bm25", "dense", "llm_rerank"]}, top_n=3),
    case("cascade_compression", "cascade", "busca", deps=E + ["compressor"], options={"stages": ["keyword", "contextual_compression"]}),
    case("custom", "custom", "python", deps=["custom"]),
]

LEGACY = [
    {"name": "legacy_tf_idf", "method": "tf_idf_retrieval", "args": ["busca densa", 2]},
    {"name": "legacy_bm25", "method": "bm25_retrieval", "args": ["python", 2]},
    {"name": "legacy_dense", "method": "dense_retrieval", "args": ["rust", 2]},
    {"name": "legacy_hybrid", "method": "hybrid_retrieval", "args": ["python busca", 3]},
    {"name": "legacy_hybrid_tfidf", "method": "hybrid_retrieval", "args": ["python busca", 3, [0.2, 0.8], "tfidf"]},
    {"name": "legacy_ensemble", "method": "ensemble_retrieval", "args": ["rust", ["dense", "keyword"], 2]},
    {"name": "legacy_rerank", "method": "rerank_retrieval", "args": ["python", 4, 2], "deps": ["reranker"]},
    {"name": "legacy_compression", "method": "contextual_compression_retrieval", "args": ["busca", 2], "deps": ["compressor"]},
    {"name": "legacy_multi_query", "method": "multi_query_retrieval", "args": ["python", 2], "deps": ["query_generator"]},
    {"name": "legacy_rephrase", "method": "rephrase_query_retrieval", "args": ["python", 2], "deps": ["llm"]},
    {"name": "legacy_reprase", "method": "reprase_query_retrieval", "args": ["python", 2], "deps": ["query_generator"]},
    {"name": "legacy_self_query", "method": "self_query_retrieval", "args": ["código rust", "docs", [], [], [], 2], "deps": ["query_constructor"]},
]


def build(deps, **config):
    values = copy.deepcopy(BASE)
    values["documents"] = [Document(**d) for d in DOCUMENTS]
    values["options"] = {}
    for k, v in config.items():
        if k == "options":
            values["options"].update(copy.deepcopy(v))
        elif k == "documents":
            values["documents"] = [Document(**d) for d in v]
        else:
            values[k] = copy.deepcopy(v)
    for name in deps:
        DEPS[name](values)
    return values


def dump(result):
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in result]


def main(path):
    out = {"documents": DOCUMENTS, "parent_store": PARENT_STORE, "base": {k: v for k, v in BASE.items() if k != "documents"}, "cases": [], "legacy": []}
    for c in CASES:
        manager = RetrievalManager(build(c["deps"], strategy=c["strategy"], **c["config"]))
        c = dict(c, expected=dump(manager.retrieve(c["query"])))
        assert c["expected"], c["name"]
        out["cases"].append(c)
    for c in LEGACY:
        deps = E + c.get("deps", [])
        manager = RetrievalManager(build(deps, strategy="dense"))
        c = dict(c, deps=deps, expected=dump(getattr(manager, c["method"])(*c["args"])))
        assert c["expected"], c["name"]
        out["legacy"].append(c)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{len(out['cases'])} casos + {len(out['legacy'])} legados -> {path}")


if __name__ == "__main__":
    main(sys.argv[1])
