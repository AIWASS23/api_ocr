//! `docling_hybrid` (nova) — o `HybridChunker` do docling (port em Rust do docling.rs): lê o
//! Markdown/HTML como documento estruturado, faz um chunk por item (parágrafo, lista inteira,
//! tabela inteira em triplas `linha, coluna = valor`), divide os grandes com o tokenizador e
//! junta vizinhos pequenos sob o mesmo título. O texto de cada chunk leva o caminho de títulos
//! (`contextualize`), como o docling recomenda para embeddings.
//!
//! Precisa da feature `docling` e de um tokenizador (`manager.load_model("docling_hybrid")`):
//! o `chunk_size` é o limite em tokens.
//!
//! Opções: `input_format` (`"markdown"` | `"html"`; padrão: `document_type` da configuração ou
//! markdown), `merge_peers` (`true`), `contextualize` (`true`). Metadados: `headings`,
//! `doc_items`.

#[cfg(feature = "docling")]
pub use imp::ChunkDoclingHybrid;

#[cfg(not(feature = "docling"))]
pub struct ChunkDoclingHybrid;

#[cfg(not(feature = "docling"))]
impl ChunkDoclingHybrid {
    pub fn new(
        _sizing: crate::chunker::Sizing,
        _tokenizer: std::sync::Arc<dyn crate::models::tokenizer::TextTokenizer>,
        _document_type: Option<&str>,
        _o: crate::core::options::Options,
    ) -> crate::error::Result<Self> {
        Err(crate::error::ChunkError::FeatureDisabled { strategy: "docling_hybrid", feature: "docling" })
    }
}

#[cfg(not(feature = "docling"))]
impl crate::chunker::Chunker for ChunkDoclingHybrid {
    fn name(&self) -> &str {
        "docling_hybrid"
    }
    fn chunk_size(&self) -> usize {
        0
    }
    fn chunk_overlap(&self) -> usize {
        0
    }
    fn chunk(&self, _text: &str) -> crate::error::Result<Vec<String>> {
        Err(crate::error::ChunkError::FeatureDisabled { strategy: "docling_hybrid", feature: "docling" })
    }
}

#[cfg(feature = "docling")]
mod imp {
    use std::sync::Arc;

    use docling::chunker::{ChunkTokenizer, HybridChunker, contextualize};
    use docling::{DocumentConverter, InputFormat, SourceDocument};

    use crate::chunker::{Chunker, Sizing};
    use crate::core::options::Options;
    use crate::core::pystr::{normalize_chunks, strip};
    use crate::document::Document;
    use crate::error::{ChunkError, Result};
    use crate::models::tokenizer::TextTokenizer;

    /// Adapta o nosso tokenizador ao trait do docling.
    struct Budget {
        tokenizer: Arc<dyn TextTokenizer>,
        max_tokens: usize,
    }

    impl ChunkTokenizer for Budget {
        fn count_tokens(&self, text: &str) -> usize {
            self.tokenizer.count(text)
        }
        fn max_tokens(&self) -> usize {
            self.max_tokens
        }
    }

    pub struct ChunkDoclingHybrid {
        sizing: Sizing,
        chunker: HybridChunker<Budget>,
        format: InputFormat,
        contextualize: bool,
    }

    impl ChunkDoclingHybrid {
        pub fn new(sizing: Sizing, tokenizer: Arc<dyn TextTokenizer>, document_type: Option<&str>, mut o: Options) -> Result<Self> {
            let format = match o.string("input_format")?.as_deref().or(document_type) {
                None | Some("markdown" | "md") => InputFormat::Md,
                Some("html") => InputFormat::Html,
                Some(other) => return Err(ChunkError::config(format!("docling_hybrid: input_format '{other}' (markdown ou html)"))),
            };
            let merge_peers = o.bool("merge_peers", true)?;
            let contextualize = o.bool("contextualize", true)?;
            o.finish()?;
            let chunker = HybridChunker::new(Budget { tokenizer, max_tokens: sizing.chunk_size }).with_merge_peers(merge_peers);
            Ok(Self { sizing, chunker, format, contextualize })
        }

        fn chunks(&self, text: &str) -> Result<Vec<docling::chunker::DocChunk>> {
            let name = if self.format == InputFormat::Html { "input.html" } else { "input.md" };
            let source = SourceDocument::from_bytes(name, self.format, text.as_bytes().to_vec());
            let result = DocumentConverter::new().convert(source).map_err(|e| ChunkError::Model(format!("docling: {e}")))?;
            Ok(self.chunker.chunk(&result.document))
        }
    }

    impl Chunker for ChunkDoclingHybrid {
        crate::sizing_accessors!("docling_hybrid");

        fn chunk(&self, text: &str) -> Result<Vec<String>> {
            Ok(self.chunk_documents(text)?.into_iter().map(|d| d.page_content).collect())
        }

        fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
            if strip(text).is_empty() {
                return Ok(Vec::new());
            }
            let mut out = Vec::new();
            for c in self.chunks(text)? {
                let body = if self.contextualize { contextualize(&c) } else { c.text.clone() };
                let Some(body) = normalize_chunks([body]).pop() else { continue };
                let mut d = Document::new(body);
                if let Some(h) = &c.headings {
                    d = d.insert("headings", serde_json::to_value(h).unwrap_or_default());
                }
                let refs: Vec<&str> = c.doc_items.iter().map(|i| i.self_ref.as_str()).collect();
                d = d.insert("doc_items", serde_json::to_value(refs).unwrap_or_default());
                out.push(d);
            }
            Ok(out)
        }
    }
}
