//! `spacy` — frases pelo `sentencizer` do spaCy sobre `spacy.blank(language)` (port em Rust
//! do tokenizador e do sentencizer, com as regras do spaCy para `pt` e `en`), juntadas com
//! `"\n\n"` e cortadas no tamanho pelo splitter recursivo (`keep_separator="end"`). Para outro
//! idioma, usa a estratégia `sentence`, como o chatbot faz quando o spaCy não carrega.
//!
//! Opções: `pipeline` (só `sentencizer`: modelos treinados do spaCy não rodam em Rust),
//! `language` (padrão: `language` da configuração ou `pt`).

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, normalize_chunks};
use crate::core::splitter::{KeepSeparator, RecursiveSplitter, TextSplitter};
use crate::error::{ChunkError, Result};
use crate::models::spacy::{SpacySentencizer, sentencizer};
use crate::strategies::sentence::ChunkSentence;

pub struct ChunkSpacy {
    sizing: Sizing,
    nlp: Option<&'static SpacySentencizer>,
    splitter: RecursiveSplitter,
    fallback: ChunkSentence,
}

impl ChunkSpacy {
    pub fn new(sizing: Sizing, pipeline: &str, language: &str, o: Options) -> Result<Self> {
        o.finish()?;
        if pipeline != "sentencizer" {
            return Err(ChunkError::config(format!(
                "spacy: pipeline '{pipeline}' não é suportado em Rust (modelos treinados do spaCy); use 'sentencizer'"
            )));
        }
        let base = TextSplitter::new(sizing.chunk_size, sizing.chunk_overlap)?.with_keep_separator(KeepSeparator::End);
        Ok(Self {
            sizing,
            nlp: sentencizer(language).ok(),
            splitter: RecursiveSplitter::new(base, &["\n\n", "\n", " ", ""], false)?,
            fallback: ChunkSentence::new(sizing)?,
        })
    }
}

impl Chunker for ChunkSpacy {
    crate::sizing_accessors!("spacy");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if pystr::strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let Some(nlp) = self.nlp else {
            return self.fallback.chunk(text);
        };
        let sentences: Vec<&str> = nlp
            .sentence_spans(text)?
            .into_iter()
            .map(|(s, e)| pystr::strip(&text[s..e]))
            .filter(|s| !s.is_empty())
            .collect();
        let joined = sentences.join("\n\n");
        Ok(normalize_chunks(self.splitter.split_text(&joined)?))
    }
}
