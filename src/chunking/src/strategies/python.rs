//! `python` — código Python (`code` com `language="python"`): `class`, `def`, métodos, blocos.

use crate::chunker::Sizing;
use crate::core::options::Options;
use crate::error::Result;
use crate::strategies::code::ChunkCode;

pub fn chunk_python(sizing: Sizing, o: Options) -> Result<ChunkCode> {
    ChunkCode::new(sizing, "python", "python", o)
}
