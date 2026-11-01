//! `siglip` — SigLIP do Google: como o CLIP (texto e imagem no mesmo espaço), treinado com perda
//! sigmoide. Imagem redimensionada direto para 224×224 (sem recorte), normalização 0,5/0,5.
//! Texto completado até 64 tokens, como no treino.
//!
//! Os repositórios `google/siglip-*` não publicam ONNX: os grafos vêm do espelho
//! `Xenova/siglip-*`.
//!
//! Padrão: `google/siglip-base-patch16-224` (768 dimensões). Diferença intencional: o Python
//! completa os textos só até o mais longo do lote (`padding=True`), o que muda o vetor do SigLIP
//! (o modelo usa a última posição); aqui vale o `padding="max_length"` recomendado.

use crate::config::EmbeddingConfig;
use crate::error::Result;
use crate::strategies::clip::{ClipEmbedder, load_dual};

pub fn load(config: &EmbeddingConfig) -> Result<ClipEmbedder> {
    load_dual(config, "siglip", 64, true)
}
