//! `line` — prefere quebras de linha, depois palavras e caracteres.

crate::natural_strategy!(ChunkLine, "line", &[r"\r?\n", r"\s+", ""]);
