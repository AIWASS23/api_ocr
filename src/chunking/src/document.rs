//! `Document` com o mesmo formato do LangChain/`data_preparation` (`page_content` + `metadata`).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type Metadata = Map<String, Value>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Document {
    pub page_content: String,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Document {
    pub fn new(page_content: impl Into<String>) -> Self {
        Self { page_content: page_content.into(), metadata: Metadata::new() }
    }

    pub fn with_metadata(page_content: impl Into<String>, metadata: Metadata) -> Self {
        Self { page_content: page_content.into(), metadata }
    }

    /// Acrescenta (ou sobrescreve) um campo de metadado.
    pub fn insert(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.to_string(), value.into());
        self
    }
}
