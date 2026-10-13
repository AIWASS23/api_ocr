//! `token_tiktoken` — janelas fixas de tokens tiktoken (`TokenTextSplitter`), encoding
//! `cl100k_base` por padrão. O BPE vem embutido no `tiktoken-rs` (funciona offline).
//!
//! Opções: `encoding_name` (`cl100k_base`), `model_name` (sobrepõe o encoding),
//! `allowed_special` (`"all"` ou lista), `disallowed_special` (`"all"` ou lista; texto com um
//! token especial proibido é recusado, como no tiktoken).

use crate::chunker::{Chunker, Sizing};
use crate::error::Result;

#[cfg(feature = "tiktoken")]
pub use imp::ChunkTikToken;

#[cfg(not(feature = "tiktoken"))]
pub struct ChunkTikToken;

#[cfg(not(feature = "tiktoken"))]
impl ChunkTikToken {
    pub fn from_options(_sizing: Sizing, _o: crate::core::options::Options) -> Result<Self> {
        Err(crate::error::ChunkError::FeatureDisabled { strategy: "token_tiktoken", feature: "tiktoken" })
    }
}

#[cfg(not(feature = "tiktoken"))]
impl Chunker for ChunkTikToken {
    fn name(&self) -> &str {
        "token_tiktoken"
    }
    fn chunk_size(&self) -> usize {
        0
    }
    fn chunk_overlap(&self) -> usize {
        0
    }
    fn chunk(&self, _text: &str) -> Result<Vec<String>> {
        Err(crate::error::ChunkError::FeatureDisabled { strategy: "token_tiktoken", feature: "tiktoken" })
    }
}

#[cfg(feature = "tiktoken")]
mod imp {
    use std::collections::HashSet;

    use serde_json::Value;
    use tiktoken_rs::CoreBPE;
    use tiktoken_rs::tokenizer::Tokenizer;

    use super::*;
    use crate::core::options::Options;
    use crate::core::pystr::{normalize_chunks, strip};
    use crate::core::tokens::split_text_on_tokens;
    use crate::error::ChunkError;

    pub struct ChunkTikToken {
        sizing: Sizing,
        bpe: &'static CoreBPE,
        allowed: HashSet<String>,
        disallowed: HashSet<String>,
    }

    fn encoding(name: &str) -> Result<&'static CoreBPE> {
        let tok = match name {
            "cl100k_base" => Tokenizer::Cl100kBase,
            "o200k_base" => Tokenizer::O200kBase,
            "o200k_harmony" => Tokenizer::O200kHarmony,
            "p50k_base" => Tokenizer::P50kBase,
            "p50k_edit" => Tokenizer::P50kEdit,
            "r50k_base" => Tokenizer::R50kBase,
            "gpt2" => Tokenizer::Gpt2,
            other => return Err(ChunkError::config(format!("Unknown encoding {other}"))),
        };
        tiktoken_rs::bpe_for_tokenizer(tok).map_err(|e| ChunkError::Model(e.to_string()))
    }

    /// `"all"`, lista de nomes ou ausente.
    fn special_set(v: Option<Value>, all: &HashSet<&str>, default_all: bool) -> Result<HashSet<String>> {
        match v {
            None => Ok(if default_all { all.iter().map(|s| s.to_string()).collect() } else { HashSet::new() }),
            Some(Value::String(s)) if s == "all" => Ok(all.iter().map(|s| s.to_string()).collect()),
            Some(Value::Array(items)) => items
                .into_iter()
                .map(|i| i.as_str().map(str::to_string).ok_or_else(|| ChunkError::config("special tokens must be strings")))
                .collect(),
            Some(other) => Err(ChunkError::config(format!("special token option must be 'all' or a list, got {other}"))),
        }
    }

    impl ChunkTikToken {
        pub fn from_options(sizing: Sizing, mut o: Options) -> Result<Self> {
            let encoding_name = o.string("encoding_name")?.unwrap_or_else(|| "cl100k_base".into());
            let model_name = o.string("model_name")?;
            let allowed = o.take::<Value>("allowed_special")?;
            let disallowed = o.take::<Value>("disallowed_special")?;
            o.ignore(&["add_start_index", "strip_whitespace", "keep_separator"]);
            o.finish()?;
            let bpe = match model_name {
                Some(m) => tiktoken_rs::bpe_for_model(&m).map_err(|e| ChunkError::config(e.to_string()))?,
                None => encoding(&encoding_name)?,
            };
            let specials = bpe.special_tokens();
            let allowed = special_set(allowed, &specials, false)?;
            let mut disallowed = special_set(disallowed, &specials, true)?;
            disallowed.retain(|s| !allowed.contains(s));
            Ok(Self { sizing, bpe, allowed, disallowed })
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            if let Some(tok) = self.disallowed.iter().filter(|t| text.contains(t.as_str())).min_by_key(|t| text.find(t.as_str())) {
                return Err(ChunkError::config(format!(
                    "Encountered text corresponding to disallowed special token '{tok}'.\n\
                     If you want this text to be encoded as a special token, pass it to `allowed_special`.\n\
                     If you want this text to be encoded as normal text, disable the check for this token \
                     by passing `disallowed_special=(enc.special_tokens_set - {{'{tok}'}})`."
                )));
            }
            let allowed: HashSet<&str> = self.allowed.iter().map(String::as_str).collect();
            let (ids, _) = self.bpe.encode(text, &allowed).map_err(|e| ChunkError::Model(e.to_string()))?;
            Ok(ids)
        }
    }

    impl Chunker for ChunkTikToken {
        crate::sizing_accessors!("token_tiktoken");

        fn chunk(&self, text: &str) -> Result<Vec<String>> {
            if strip(text).is_empty() {
                return Ok(Vec::new());
            }
            let ids = self.encode(text)?;
            let chunks = split_text_on_tokens(&ids, self.sizing.chunk_size, self.sizing.chunk_overlap, |w| {
                let bytes = self.bpe.decode_bytes(w).map_err(|e| ChunkError::Model(e.to_string()))?;
                // `errors="replace"` do tiktoken.
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            })?;
            Ok(normalize_chunks(chunks))
        }
    }
}
