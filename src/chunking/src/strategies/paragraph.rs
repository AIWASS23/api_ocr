//! `paragraph` — prefere parágrafos (linha em branco), depois frases, palavras e caracteres.

crate::natural_strategy!(ChunkParagraph, "paragraph", &[r"\n\s*\n", r"(?<=[.!?…])\s+", r"\s+", ""]);
