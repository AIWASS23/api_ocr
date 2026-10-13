//! `nltk` — frases pelo Punkt do NLTK (port em Rust, com os parâmetros pré-treinados do
//! `punkt_tab`), juntadas com `"\n\n"` até o tamanho (`NLTKTextSplitter`). Sem os dados do
//! `punkt_tab` para o idioma, usa a estratégia `sentence`, como o chatbot faz no `LookupError`.
//!
//! Opções: `language` (padrão: `language` da configuração ou `portuguese`), `separator`
//! (`"\n\n"`), `use_span_tokenize` (`false`), `nltk_data` (diretório extra de busca).

use std::path::PathBuf;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{normalize_chunks, strip};
use crate::core::splitter::TextSplitter;
use crate::error::{ChunkError, Result};
use crate::models::punkt::PunktTokenizer;
use crate::strategies::sentence::ChunkSentence;

pub struct ChunkNltk {
    sizing: Sizing,
    punkt: Option<PunktTokenizer>,
    base: TextSplitter,
    separator: String,
    use_span_tokenize: bool,
    fallback: ChunkSentence,
}

impl ChunkNltk {
    pub fn new(sizing: Sizing, language: &str, mut o: Options) -> Result<Self> {
        let separator = o.string("separator")?.unwrap_or_else(|| "\n\n".to_string());
        let use_span_tokenize = o.bool("use_span_tokenize", false)?;
        let data = o.string("nltk_data")?.map(PathBuf::from);
        let strip_ws = o.bool("strip_whitespace", true)?;
        o.ignore(&["add_start_index"]);
        o.finish()?;
        if use_span_tokenize && !separator.is_empty() {
            return Err(ChunkError::config("When use_span_tokenize is True, separator should be ''"));
        }
        let punkt = PunktTokenizer::load(language, data.as_deref()).ok();
        Ok(Self {
            sizing,
            punkt,
            base: TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_strip_whitespace(strip_ws),
            separator,
            use_span_tokenize,
            fallback: ChunkSentence::new(sizing)?,
        })
    }

    /// `true` se os parâmetros do Punkt foram encontrados (senão, usa `sentence`).
    pub fn has_punkt(&self) -> bool {
        self.punkt.is_some()
    }
}

impl Chunker for ChunkNltk {
    crate::sizing_accessors!("nltk");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let Some(punkt) = &self.punkt else {
            return self.fallback.chunk(text);
        };
        let splits: Vec<String> = if self.use_span_tokenize {
            // Cada frase leva consigo o espaço que a separa da anterior.
            let spans = punkt.span_tokenize(text);
            spans
                .iter()
                .enumerate()
                .map(|(i, &(s, e))| if i > 0 { text[spans[i - 1].1..e].to_string() } else { text[s..e].to_string() })
                .collect()
        } else {
            punkt.tokenize(text).into_iter().map(str::to_string).collect()
        };
        Ok(normalize_chunks(self.base.merge_splits(&splits, &self.separator)))
    }
}
