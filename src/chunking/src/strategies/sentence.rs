//! `sentence` — prefere fronteiras de frase (`.`, `!`, `?`, `…` seguidos de aspas/parênteses
//! opcionais e espaço), depois quebras de linha, palavras e caracteres.

crate::natural_strategy!(ChunkSentence, "sentence", &[r#"(?<=[.!?…])(?:["'’”»)\]]*)\s+"#, r"\n+", r"\s+", ""]);
