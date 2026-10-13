//! Contrato comum a todas as estratégias e comportamento das 8 estratégias novas (que não têm
//! equivalente no Python). Modelos são substituídos por dublês determinísticos.

#![cfg(all(feature = "hf-tokenizers", feature = "html", feature = "tiktoken"))]

use std::sync::Arc;

use chunking::models::embeddings::Embedder;
use chunking::models::llm::ContextModel;
use chunking::models::token_embeddings::{TokenEmbedder, TokenEmbeddings};
use chunking::models::tokenizer::HfTokenizer;
use chunking::{ChunkError, ChunkManager, Chunker, ChunkerAdapter, Document, Metadata, SUPPORTED_STRATEGIES, Sizing};
use serde_json::{Value, json};

fn minilm() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minilm").to_string_lossy().into_owned()
}

struct Letters;
impl Embedder for Letters {
    fn embed_documents(&self, texts: &[String]) -> chunking::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![1f32; 27];
                for c in t.to_lowercase().chars().filter(char::is_ascii_lowercase) {
                    v[(c as u8 - b'a') as usize] += 1.0;
                }
                v
            })
            .collect())
    }
}

/// LLM falso: "Contexto: <primeira palavra do chunk>".
struct FakeLlm {
    fail: bool,
}
impl ContextModel for FakeLlm {
    fn situate(&self, document: &str, chunk: &str) -> chunking::Result<String> {
        if self.fail {
            return Err(ChunkError::Llm("fora do ar".into()));
        }
        assert!(document.len() >= chunk.len() || document.contains(&chunk[..chunk.len().min(10)]));
        Ok(format!("Contexto: {}", chunk.split_whitespace().next().unwrap_or("")))
    }
}

/// Embeddings por token falsos: um token por palavra; vetor = [posição, 1].
struct WordVectors;
impl TokenEmbedder for WordVectors {
    fn embed_tokens(&self, text: &str) -> chunking::Result<TokenEmbeddings> {
        let mut out = TokenEmbeddings::default();
        let mut start = None;
        for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ' '))) {
            match (start, c.is_whitespace()) {
                (None, false) => start = Some(i),
                (Some(s), true) => {
                    out.offsets.push((s, i));
                    out.vectors.push(vec![out.vectors.len() as f32, 1.0]);
                    start = None;
                }
                _ => {}
            }
        }
        Ok(out)
    }
    fn model_name(&self) -> &str {
        "word-vectors"
    }
}

fn manager(strategy: &str, size: usize, overlap: usize, options: Value) -> ChunkManager {
    let m = ChunkManager::from_value(json!({
        "strategy": strategy, "chunk_size": size, "chunk_overlap": overlap,
        "model_name": minilm(), "options": options,
    }))
    .unwrap();
    m.with_tokenizer(Arc::new(HfTokenizer::load(&minilm()).unwrap()))
        .with_embedder(Arc::new(Letters))
        .with_context_model(Arc::new(FakeLlm { fail: false }))
        .with_token_embedder(Arc::new(WordVectors))
}

const PROSE: &str = "# Guia\n\nO município cobra IPTU anualmente. O Sr. João explicou as regras.\n\n## Taxas\n\nA taxa de lixo é mensal. Pagamentos atrasados têm multa de 2%.\n\nArt. 1º Fica instituída a taxa.\n§ 1º A taxa é anual.\n\nPergunta: Como pagar?\nResposta: No portal ou no banco.";
const HTML: &str = "<html><body><h1>Título</h1><p>Primeiro parágrafo do documento.</p><h2>Seção</h2><p>Segundo parágrafo.</p></body></html>";
const JSONDOC: &str = r#"{"a": 1, "b": {"c": "texto"}}"#;

#[test]
fn every_strategy_honours_the_contract() {
    for &strategy in SUPPORTED_STRATEGIES {
        if strategy == "docling_hybrid" && !cfg!(feature = "docling") {
            let err = manager(strategy, 200, 20, json!({})).chunk("x").unwrap_err();
            assert!(matches!(err, ChunkError::FeatureDisabled { .. }), "{strategy}: {err}");
            continue;
        }
        let input = match strategy {
            "html_header" | "html_section" | "html_semantic" | "html" => HTML,
            "json" => JSONDOC,
            _ => PROSE,
        };
        let size = if strategy.starts_with("token") || strategy == "semchunk" || strategy == "docling_hybrid" { 40 } else { 200 };
        let m = manager(strategy, size, 5, json!({}));
        let c = m.build(None).unwrap_or_else(|e| panic!("{strategy}: {e}"));
        assert_eq!(c.name(), strategy);
        // Vazio e só espaços: nenhum chunk.
        assert!(c.chunk("").unwrap().is_empty(), "{strategy}");
        assert!(c.chunk("  \n\t ").unwrap().is_empty(), "{strategy}");
        // Chunks normalizados.
        let chunks = c.chunk(input).unwrap_or_else(|e| panic!("{strategy}: {e}"));
        assert!(!chunks.is_empty(), "{strategy} não gerou chunks");
        for ch in &chunks {
            assert!(!ch.is_empty() && ch.trim() == ch, "{strategy}: chunk não normalizado {ch:?}");
        }
        assert_eq!(c.split_text(input).unwrap(), chunks, "{strategy}: split_text == chunk");
        // Metadados de origem preservados nos documentos.
        let mut meta = Metadata::new();
        meta.insert("source".into(), "guia.pdf".into());
        let docs = c.split_documents(&[Document::with_metadata(input, meta.clone())]).unwrap();
        assert_eq!(docs.len(), chunks.len(), "{strategy}");
        assert!(docs.iter().all(|d| d.metadata["source"] == "guia.pdf"), "{strategy}");
        assert_eq!(docs.iter().map(|d| d.page_content.clone()).collect::<Vec<_>>(), chunks, "{strategy}");
        let created = c.create_documents(&[input, input], Some(&[meta.clone(), meta])).unwrap();
        assert_eq!(created.len(), 2 * chunks.len(), "{strategy}");
        assert!(c.create_documents(&[input], Some(&[])).is_err(), "{strategy}: tamanhos diferentes");
        assert_eq!(c.batch(&[input, ""]).unwrap(), vec![chunks.clone(), vec![]], "{strategy}");
    }
}

#[test]
fn manager_behaves_like_python() {
    let m = ChunkManager::from_value(json!({"strategy": "fixed_size", "chunk_size": 50})).unwrap();
    assert_eq!(m.config().strategy, "character");
    assert!(!m.requires_model(None).unwrap());
    assert!(m.requires_model(Some("semantic")).unwrap());
    // Estratégia que precisa de modelo sem modelo carregado: erro explícito.
    let err = m.chunk_with("texto", Some("token")).unwrap_err();
    assert!(matches!(err, ChunkError::ModelNotLoaded(_)), "{err}");
    // Troca de estratégia numa chamada.
    assert_eq!(m.chunk_with("Uma frase. Outra frase.", Some("sentence")).unwrap(), ["Uma frase. Outra frase."]);
    // Aliases legados.
    assert_eq!(m.chunks_recursive("a b").unwrap(), ["a b"]);
    // Splitter externo.
    let ext = ChunkerAdapter::new("pipes", Sizing::new(10, 0).unwrap(), |t| Ok(t.split('|').map(String::from).collect()));
    let m = m.with_chunker(Arc::new(ext));
    assert_eq!(m.chunk("a| b |c").unwrap(), ["a", "b", "c"]);
}

#[test]
fn markdown_header_contextual_repeats_the_breadcrumb_in_every_piece() {
    let long = format!("# Lei Tributária\n\n## Capítulo II\n\n### Taxas\n\n{}", "A taxa é devida mensalmente. ".repeat(20));
    let m = manager("markdown_header_contextual", 160, 0, json!({"title": "Código Municipal"}));
    let docs = m.chunk_documents(&long).unwrap();
    assert!(docs.len() > 2, "seção longa deveria ser subdividida");
    for d in &docs {
        assert!(d.page_content.starts_with("Código Municipal > Lei Tributária > Capítulo II > Taxas\n\n"), "{:?}", d.page_content);
        assert!(d.page_content.chars().count() <= 160, "{}", d.page_content.chars().count());
        assert_eq!(d.metadata["h3"], "Taxas");
    }
    // Caminho maior que metade do tamanho: títulos externos omitidos.
    let m = manager("markdown_header_contextual", 60, 0, json!({"title": "Um título de documento bem comprido mesmo"}));
    let d = &m.chunk_documents("# Capítulo\n\nTexto curto.").unwrap()[0];
    assert!(d.page_content.starts_with("… > Capítulo\n\n"), "{:?}", d.page_content);
}

#[test]
fn table_keeps_rows_with_their_header() {
    let mut text = String::from("Introdução ao quadro.\n\nTabela 1 - Horários\n\n| Unidade | Horário |\n|---|---|\n");
    for i in 0..30 {
        text.push_str(&format!("| Unidade {i} | 8h às 17h |\n"));
    }
    text.push_str("\nTexto depois da tabela.");
    let m = manager("table", 200, 0, json!({}));
    let docs = m.chunk_documents(&text).unwrap();
    let tables: Vec<&Document> = docs.iter().filter(|d| d.metadata["kind"] == "table").collect();
    assert!(tables.len() > 1, "tabela grande deveria virar vários pedaços");
    for t in &tables {
        assert!(t.page_content.starts_with("Tabela 1 - Horários\n| Unidade | Horário |\n|---|---|\n"), "{:?}", t.page_content);
        assert!(t.page_content.chars().count() <= 200);
        assert_eq!(t.metadata["caption"], "Tabela 1 - Horários");
    }
    let rows: usize = tables.iter().map(|t| t.page_content.lines().filter(|l| l.starts_with("| Unidade ") && l.contains("8h")).count()).sum();
    assert_eq!(rows, 30, "nenhuma linha perdida ou duplicada");
    assert!(docs.iter().any(|d| d.page_content == "Texto depois da tabela."));
    assert!(!docs.iter().any(|d| d.metadata["kind"] == "text" && d.page_content.contains("Tabela 1")), "legenda não fica no texto");
    // Formato chave: valor.
    let m = manager("table", 500, 0, json!({"format": "key_value"}));
    let kv = m.chunk("| Dia | Horário |\n|:---|---:|\n| Seg | 8h |\n").unwrap();
    assert_eq!(kv, ["Dia: Seg; Horário: 8h"]);
}

const LEI: &str = "LEI Nº 1.234, DE 10 DE MARÇO DE 2020\n\nDispõe sobre taxas municipais.\n\nTÍTULO I\nDAS DISPOSIÇÕES GERAIS\n\nArt. 1º Esta lei institui a taxa de coleta.\n\nCAPÍTULO II - DA COBRANÇA\n\nSeção I\nDo Lançamento\n\nArt. 2º O lançamento é anual.\nParágrafo único. O contribuinte será notificado.\n\nArt. 3º São isentos:\nI - os templos;\nII - as entidades assistenciais;\na) com certificado;\nb) sem fins lucrativos.\n";

#[test]
fn legal_makes_one_chunk_per_article_with_its_path() {
    let m = manager("legal", 400, 0, json!({}));
    let docs = m.chunk_documents(LEI).unwrap();
    let arts: Vec<&Document> = docs.iter().filter(|d| d.metadata.contains_key("artigo")).collect();
    assert_eq!(arts.iter().map(|d| d.metadata["artigo"].as_str().unwrap()).collect::<Vec<_>>(), ["Art. 1º", "Art. 2º", "Art. 3º"]);
    let art2 = arts[1];
    assert!(art2.page_content.starts_with(
        "LEI Nº 1.234, DE 10 DE MARÇO DE 2020 > TÍTULO I - DAS DISPOSIÇÕES GERAIS > CAPÍTULO II - DA COBRANÇA > Seção I - Do Lançamento\nArt. 2º"
    ), "{:?}", art2.page_content);
    assert!(art2.page_content.contains("Parágrafo único"), "parágrafo fica com o artigo");
    assert_eq!(art2.metadata["capitulo"], "CAPÍTULO II - DA COBRANÇA");
    assert!(!arts[0].metadata.contains_key("capitulo"), "Art. 1º vem antes do capítulo");
    // Artigo maior que o tamanho: dividido nos incisos, repetindo o rótulo.
    let m = manager("legal", 60, 0, json!({"breadcrumb": false}));
    let docs = m.chunk_documents(LEI).unwrap();
    let a3: Vec<&Document> = docs.iter().filter(|d| d.metadata.get("artigo").and_then(Value::as_str) == Some("Art. 3º")).collect();
    assert!(a3.len() > 1);
    assert!(a3[1..].iter().all(|d| d.page_content.starts_with("Art. 3º (continuação)\n")));
    assert!(docs.iter().all(|d| d.page_content.chars().count() <= 60), "{docs:?}");
    // Sem artigos: corte recursivo.
    assert_eq!(manager("legal", 100, 0, json!({})).chunk("Texto comum sem artigos.").unwrap(), ["Texto comum sem artigos."]);
}

#[test]
fn qa_pairs_questions_with_answers() {
    let faq = "Perguntas frequentes da prefeitura.\n\nPergunta: Como emitir a 2ª via do IPTU?\nResposta: Pelo portal, com o número de inscrição.\n\nP: Qual o prazo?\nR: Até 31 de março.\n";
    let m = manager("qa", 300, 0, json!({}));
    let docs = m.chunk_documents(faq).unwrap();
    assert_eq!(docs[0].page_content, "Perguntas frequentes da prefeitura.");
    assert_eq!(docs[1].page_content, "Pergunta: Como emitir a 2ª via do IPTU?\nResposta: Pelo portal, com o número de inscrição.");
    assert_eq!(docs[2].page_content, "Pergunta: Qual o prazo?\nResposta: Até 31 de março.");
    assert_eq!(docs[2].metadata["qa_index"], 1);
    // Perguntas implícitas (títulos) e resposta longa dividida repetindo a pergunta.
    let md = format!("## Quem tem direito à isenção?\n\n{}\n\n## Onde pedir?\n\nNo protocolo.", "Aposentados com renda baixa. ".repeat(12));
    let docs = manager("qa", 120, 0, json!({})).chunk_documents(&md).unwrap();
    let isencao: Vec<&Document> = docs.iter().filter(|d| d.metadata["question"] == "Quem tem direito à isenção?").collect();
    assert!(isencao.len() > 1);
    assert!(isencao.iter().all(|d| d.page_content.starts_with("Pergunta: Quem tem direito à isenção?\nResposta: ")));
    assert!(docs.iter().all(|d| d.page_content.chars().count() <= 120));
    assert_eq!(docs.last().unwrap().page_content, "Pergunta: Onde pedir?\nResposta: No protocolo.");
}

#[test]
fn parent_child_links_small_chunks_to_their_section() {
    let text = format!("# A\n\n{}\n\n# B\n\n{}", "alfa beta gama delta. ".repeat(15), "um dois três quatro. ".repeat(15));
    let m = manager("parent_child", 80, 0, json!({"parent_chunk_size": 400, "include_parent_content": true}));
    let c = m.build(None).unwrap();
    let children = c.chunk_documents(&text).unwrap();
    assert!(children.len() > 4);
    for ch in &children {
        assert!(ch.page_content.chars().count() <= 80);
        let parent = ch.metadata["parent_content"].as_str().unwrap();
        assert!(parent.contains(&ch.page_content), "filho precisa estar dentro do pai");
        assert!(parent.chars().count() <= 400);
        assert!(ch.metadata["parent_id"].as_str().unwrap().starts_with('p'));
    }
    // Pais separados por seção de Markdown (h1 no metadado).
    assert!(children.iter().any(|c| c.metadata["h1"] == "A") && children.iter().any(|c| c.metadata["h1"] == "B"));
    // Ids estáveis.
    assert_eq!(c.chunk_documents(&text).unwrap()[0].metadata["parent_id"], children[0].metadata["parent_id"]);
}

#[test]
fn contextual_retrieval_prefixes_llm_context() {
    let m = manager("contextual_retrieval", 60, 0, json!({"base_strategy": "sentence"}));
    let docs = m.chunk_documents("Primeira frase do texto. Segunda frase do texto. Terceira frase final aqui.").unwrap();
    assert!(docs.len() >= 2);
    for d in &docs {
        let orig = d.metadata["original_chunk"].as_str().unwrap();
        let ctx = d.metadata["context"].as_str().unwrap();
        assert_eq!(d.page_content, format!("{ctx}\n\n{orig}"));
        assert_eq!(ctx, format!("Contexto: {}", orig.split_whitespace().next().unwrap()));
    }
    // LLM fora do ar: erro, ou chunk sem contexto com fail_open.
    let failing = |opts: Value| {
        ChunkManager::from_value(json!({"strategy": "contextual", "chunk_size": 60, "options": opts}))
            .unwrap()
            .with_context_model(Arc::new(FakeLlm { fail: true }))
    };
    assert!(failing(json!({})).chunk("Uma frase.").is_err());
    assert_eq!(failing(json!({"fail_open": true})).chunk("Uma frase.").unwrap(), ["Uma frase."]);
}

#[test]
fn late_chunking_pools_document_level_token_vectors() {
    let text = "um dois três. quatro cinco seis. sete oito nove.";
    let m = manager("late_chunking", 20, 0, json!({"base_strategy": "sentence", "normalize": false}));
    let docs = m.chunk_documents(text).unwrap();
    assert_eq!(docs.iter().map(|d| d.page_content.as_str()).collect::<Vec<_>>(), ["um dois três.", "quatro cinco seis.", "sete oito nove."]);
    // Vetor = média dos tokens [posição, 1]: posições 0..2, 3..5, 6..8 -> médias 1, 4, 7.
    let first: Vec<f64> = serde_json::from_value(docs[1].metadata["embedding"].clone()).unwrap();
    assert_eq!(first, [4.0, 1.0]);
    assert_eq!(docs[2].metadata["start_index"], 33);
    assert_eq!(docs[2].metadata["embedding_model"], "word-vectors");
    // Normalizado por padrão.
    let docs = manager("late_chunking", 20, 0, json!({"base_strategy": "sentence"})).chunk_documents(text).unwrap();
    let v: Vec<f64> = serde_json::from_value(docs[0].metadata["embedding"].clone()).unwrap();
    assert!((v.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-6);
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_contract_preserves_order() {
    let c: Arc<dyn Chunker> = manager("sentence", 20, 0, json!({})).build(None).unwrap();
    let texts: Vec<String> = (0..20).map(|i| format!("Frase {i}. Outra frase {i}.")).collect();
    let out = chunking::asynch::abatch(c.clone(), texts.clone()).await.unwrap();
    for (i, chunks) in out.iter().enumerate() {
        assert_eq!(chunks[0], format!("Frase {i}."));
    }
    assert_eq!(chunking::asynch::achunk(c, texts[3].clone()).await.unwrap(), out[3]);
}
