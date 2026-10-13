use serde_json::{Map, Value};

/// Equivalente ao `langchain_core.documents.Document`.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub page_content: String,
    pub metadata: Map<String, Value>,
}

impl Document {
    pub fn new(page_content: impl Into<String>, metadata: Map<String, Value>) -> Self {
        Self {
            page_content: page_content.into(),
            metadata,
        }
    }
}
