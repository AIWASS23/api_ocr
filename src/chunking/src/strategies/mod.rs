//! Uma estratégia por arquivo. As 25 primeiras são o port do `src/chunking` do chatbot; as 8
//! últimas são novas. `natural.rs` é só a base comum de word/sentence/paragraph/line.

pub mod natural;

// Tamanho fixo
pub mod character;
pub mod recursive;
// Linguística
pub mod line;
pub mod nltk;
pub mod paragraph;
pub mod sentence;
pub mod spacy;
pub mod word;
// Tokens
pub mod semchunk;
pub mod token;
pub mod token_sentence_transformer;
pub mod token_tiktoken;
// Semântica
pub mod semantic;
// Documentos
pub mod html;
#[cfg(feature = "html")]
pub mod html_header;
#[cfg(feature = "html")]
pub mod html_section;
#[cfg(feature = "html")]
pub mod html_semantic;
pub mod json;
pub mod latex;
pub mod markdown;
pub mod markdown_header;
pub mod markdown_syntax;
// Código
pub mod code;
pub mod javascript;
pub mod python;
// Novas
pub mod contextual_retrieval;
pub mod docling_hybrid;
pub mod late_chunking;
pub mod legal;
pub mod markdown_header_contextual;
pub mod parent_child;
pub mod qa;
pub mod table;
