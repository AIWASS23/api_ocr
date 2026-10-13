//! `html` — HTML como código (`code` com `language="html"`): corta antes de `<body`, `<div`,
//! `<p`, `<table`, `<li`... sem interpretar o DOM. Para cortes por título use `html_header`,
//! `html_section` ou `html_semantic`.

use crate::chunker::Sizing;
use crate::core::options::Options;
use crate::error::Result;
use crate::strategies::code::ChunkCode;

pub fn chunk_html(sizing: Sizing, o: Options) -> Result<ChunkCode> {
    ChunkCode::new(sizing, "html", "html", o)
}
