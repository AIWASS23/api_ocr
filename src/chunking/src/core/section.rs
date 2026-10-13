//! Bloco de texto com o caminho de títulos que o envolve, usado pelas estratégias que dividem
//! por seções (Markdown, HTML, pai-filho).

use crate::document::Metadata;

/// Um bloco com os títulos que o envolvem, na ordem em que foram abertos (o Python mantém a
/// ordem de inserção do dicionário; o `Map` do serde_json, não).
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub content: String,
    pub headers: Vec<(String, String)>,
}

impl Section {
    pub fn metadata(&self) -> Metadata {
        self.headers.iter().map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone()))).collect()
    }
}

/// `_document_texts(include_metadata=True)`: `"v1 > v2\nconteúdo"`.
pub fn with_heading(section: &Section) -> String {
    let heading: Vec<&str> = section.headers.iter().map(|(_, v)| v.as_str()).filter(|v| !v.is_empty()).collect();
    if heading.is_empty() { section.content.clone() } else { format!("{}\n{}", heading.join(" > "), section.content) }
}

