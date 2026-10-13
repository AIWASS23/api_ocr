//! `split_text_on_tokens` do LangChain: janelas fixas de ids de token com sobreposição.

use crate::error::{ChunkError, Result};

pub fn split_text_on_tokens(
    input_ids: &[u32],
    tokens_per_chunk: usize,
    chunk_overlap: usize,
    decode: impl Fn(&[u32]) -> Result<String>,
) -> Result<Vec<String>> {
    if tokens_per_chunk <= chunk_overlap {
        return Err(ChunkError::config("tokens_per_chunk must be greater than chunk_overlap"));
    }
    let mut splits = Vec::new();
    let mut start = 0;
    while start < input_ids.len() {
        let cur = (start + tokens_per_chunk).min(input_ids.len());
        let ids = &input_ids[start..cur];
        if ids.is_empty() {
            break;
        }
        let decoded = decode(ids)?;
        if !decoded.is_empty() {
            splits.push(decoded);
        }
        if cur == input_ids.len() {
            break;
        }
        start += tokens_per_chunk - chunk_overlap;
    }
    Ok(splits)
}
