//! `word` — prefere fronteiras de palavra e corta palavras gigantes por caractere.

crate::natural_strategy!(ChunkWord, "word", &[r"\s+", ""]);
