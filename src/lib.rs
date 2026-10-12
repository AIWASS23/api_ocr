//! Port em Rust do `DataPreparation` (Docling + LangChain `Document`).

mod data_preparation;
mod document;
mod error;
mod files;
mod scheduler;
mod vlm;

pub use data_preparation::{DataPreparation, DataPreparationConfig, VlmProvider, pdf_page_count, warm_up_models};
pub use document::Document;
pub use error::{FileError, PrepError};
pub use files::{SUPPORTED_EXTENSIONS, detects_folder_or_file};
pub use scheduler::{DEFAULT_HEAVY_BYTES, DEFAULT_HEAVY_PAGES, ParallelConfig, Weight, classify};
pub use vlm::{DEFAULT_PROMPT, VlmClient};
