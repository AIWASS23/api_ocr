use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::Duration;

use docling::{DocumentConverter, Node, SourceDocument, Table};
use serde_json::{Map, Value, json};

use crate::document::Document;
use crate::error::{FileError, PrepError};
use crate::files::detects_folder_or_file;
use crate::scheduler::{ParallelConfig, Weight, classify, run_parallel};
use crate::vlm::{DEFAULT_PROMPT, VlmClient};

/// Origem da descrição de figuras (equivalente a `provider_vlm`).
#[derive(Debug, Clone, Default)]
pub enum VlmProvider {
    /// Não descreve figuras.
    #[default]
    None,
    /// Serviço hospedado compatível com OpenAI.
    Remote {
        api_key: Option<String>,
        base_url: String,
        model: String,
    },
    /// Servidor local (Ollama, LM Studio...).
    Local { base_url: String, model: String },
}

#[derive(Debug, Clone)]
pub struct DataPreparationConfig {
    pub prompt: String,
    pub provider: VlmProvider,
    /// Inclui a descrição das figuras no texto extraído.
    pub include_picture_descriptions: bool,
    /// Como distribuir vários arquivos entre threads.
    pub parallel: ParallelConfig,
    /// Tempo máximo por documento no pipeline de PDF/imagem (para entre páginas).
    pub document_timeout: Option<Duration>,
    /// Converte só as N primeiras páginas de cada documento.
    pub max_pages: Option<usize>,
}

impl Default for DataPreparationConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_PROMPT.to_string(),
            provider: VlmProvider::None,
            include_picture_descriptions: true,
            parallel: ParallelConfig::default(),
            document_timeout: None,
            max_pages: None,
        }
    }
}

pub struct DataPreparation {
    converter: DocumentConverter,
    vlm: Option<VlmClient>,
    include_picture_descriptions: bool,
    parallel: ParallelConfig,
}

impl DataPreparation {
    pub fn new(config: DataPreparationConfig) -> Result<Self, PrepError> {
        let vlm = match &config.provider {
            VlmProvider::None => None,
            VlmProvider::Remote {
                api_key,
                base_url,
                model,
            } => Some(VlmClient::new(
                base_url,
                model,
                api_key.clone(),
                &config.prompt,
            )?),
            VlmProvider::Local { base_url, model } => {
                Some(VlmClient::new(base_url, model, None, &config.prompt)?)
            }
        };

        // Imagens em escala 2.0, como `images_scale = 2.0` no Python.
        // O dispositivo (CUDA) é escolhido pela feature `cuda` + `DOCLING_RS_EP`.
        // Nunca resolve referências externas de imagem (rede/arquivos locais):
        // evita SSRF e leitura de arquivos ao converter entrada não confiável.
        let mut converter = DocumentConverter::new()
            .images_scale(2.0)
            .image_sources(docling::ImageSources::None)
            .document_timeout(config.document_timeout);
        if let Some(n) = config.max_pages.filter(|&n| n > 0) {
            converter = converter.page_range(1, n);
        }

        Ok(Self {
            converter,
            vlm,
            include_picture_descriptions: config.include_picture_descriptions,
            parallel: config.parallel,
        })
    }

    /// Converte um arquivo (ou todos de uma pasta) e devolve os `Document`s
    /// e os erros por arquivo.
    pub fn process_files(
        &self,
        folder_or_file: impl AsRef<Path>,
        format_output: &str,
    ) -> Result<(Vec<Document>, Vec<FileError>), PrepError> {
        if !matches!(format_output, "txt" | "markdown") {
            return Err(PrepError::InvalidFormat(format_output.to_string()));
        }

        let files = detects_folder_or_file(folder_or_file);

        // Leves em paralelo; pesados um por vez (o docling já usa vários
        // workers dentro de cada um). O resultado segue a ordem de entrada.
        let (light, heavy): (Vec<usize>, Vec<usize>) = (0..files.len())
            .partition(|&i| classify(&files[i], &self.parallel) == Weight::Light);

        let mut results: Vec<Option<Result<Vec<Document>, FileError>>> =
            files.iter().map(|_| None).collect();

        let light_files: Vec<&PathBuf> = light.iter().map(|&i| &files[i]).collect();
        let out = run_parallel(&light_files, self.parallel.light_workers(), |p| {
            self.process_guarded(p)
        });
        for (&i, r) in light.iter().zip(out) {
            results[i] = Some(r);
        }
        for &i in &heavy {
            results[i] = Some(self.process_guarded(&files[i]));
        }

        let mut docs = Vec::new();
        let mut errors = Vec::new();
        for r in results.into_iter().flatten() {
            match r {
                Ok(mut d) => docs.append(&mut d),
                Err(e) => errors.push(e),
            }
        }

        Ok((docs, errors))
    }

    /// `process_one` com erro/pânico convertidos em `FileError`.
    fn process_guarded(&self, path: &Path) -> Result<Vec<Document>, FileError> {
        let source_path = path.to_string_lossy().into_owned();
        let outcome = catch_unwind(AssertUnwindSafe(|| self.process_one(path, &source_path)))
            .unwrap_or_else(|_| Err(PrepError::Panic));
        outcome.map_err(|e| {
            eprintln!("[ERRO] Falha ao converter {source_path}: {e}");
            FileError {
                file: source_path,
                error: e.to_string(),
            }
        })
    }

    fn process_one(&self, path: &Path, source_path: &str) -> Result<Vec<Document>, PrepError> {
        let mut result = self.converter.convert(SourceDocument::from_file(path)?)?;

        if let Some(vlm) = &self.vlm {
            describe_pictures(&mut result.document.nodes, vlm);
        }

        let mut extracted = Extracted::default();
        for node in &result.document.nodes {
            extracted.visit(node, self.include_picture_descriptions);
        }

        let mut docs = Vec::new();
        if let Some(doc) = build_text_document(&extracted.text, source_path) {
            docs.push(doc);
        }
        docs.extend(build_table_documents(&extracted.tables, source_path));
        Ok(docs)
    }
}

/// Preenche `description` de cada figura que ainda não tem uma.
/// Falhas numa figura não derrubam o arquivo.
fn describe_pictures(nodes: &mut [Node], vlm: &VlmClient) {
    for node in nodes {
        match node {
            Node::Picture {
                image: Some(image),
                description,
                ..
            } if description.is_none() => match vlm.describe(&image.data_uri()) {
                Ok(text) => {
                    *description = Some(docling_description(text));
                }
                Err(e) => eprintln!("[AVISO] Falha ao descrever figura: {e}"),
            },
            Node::Group { children, .. } | Node::PictureChildren(children) => {
                describe_pictures(children, vlm)
            }
            Node::Located { inner, .. }
            | Node::Prov { inner, .. }
            | Node::Track { inner, .. }
            | Node::Commented { inner, .. }
            | Node::Furniture { inner, .. } => {
                describe_pictures(std::slice::from_mut(inner.as_mut()), vlm)
            }
            _ => {}
        }
    }
}

fn docling_description(text: String) -> docling_core::PictureDescription {
    docling_core::PictureDescription {
        text,
        provenance: "vlm".to_string(),
    }
}

/// Texto corrido e tabelas coletados numa única passada pela árvore.
/// Ao contrário do Docling Python, as tabelas já são nós separados do texto,
/// então não há filtragem por bbox.
#[derive(Default)]
struct Extracted<'a> {
    text: Vec<String>,
    tables: Vec<(&'a Table, Option<&'a str>)>,
}

impl<'a> Extracted<'a> {
    fn push_text(&mut self, text: &str) {
        let text = text.trim();
        if !text.is_empty() {
            self.text.push(text.to_string());
        }
    }

    fn visit(&mut self, node: &'a Node, pictures: bool) {
        match node {
            Node::Heading { text, .. }
            | Node::Paragraph { text }
            | Node::CheckboxItem { text, .. }
            | Node::ListItem { text, .. }
            | Node::Code { text, .. }
            | Node::Caption { text, .. }
            | Node::LabeledText { text, .. }
            | Node::TextDump(text) => self.push_text(text),
            Node::Formula { latex, orig, .. } => {
                self.push_text(if latex.trim().is_empty() { orig } else { latex })
            }
            Node::InlineGroup { md_text, .. } => self.push_text(md_text),
            Node::Table(table) => self.tables.push((table, None)),
            Node::Chart { table, caption, .. } => {
                if let Some(c) = caption {
                    self.push_text(c);
                }
                self.tables.push((table, None));
            }
            Node::Picture {
                caption,
                description,
                ..
            } => {
                if let Some(c) = caption {
                    self.push_text(c);
                }
                if pictures && let Some(d) = description {
                    self.push_text(&d.text);
                }
            }
            Node::FieldRegion { items } => {
                for item in items {
                    let line = [&item.key, &item.value]
                        .into_iter()
                        .flatten()
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join(": ");
                    self.push_text(&line);
                }
            }
            Node::KeyValueGraph { cells, .. } => {
                for cell in cells {
                    self.push_text(&cell.text);
                }
            }
            Node::Group { children, .. } | Node::PictureChildren(children) => {
                for child in children {
                    self.visit(child, pictures);
                }
            }
            Node::Located { inner, .. }
            | Node::Prov { inner, .. }
            | Node::Track { inner, .. }
            | Node::Commented { inner, .. } => self.visit(inner, pictures),
            // Cabeçalhos/rodapés, comentários, quebras de página etc. ficam de fora.
            _ => {}
        }
    }
}

fn build_text_document(parts: &[String], source_path: &str) -> Option<Document> {
    if parts.is_empty() {
        return None;
    }
    let mut metadata = Map::new();
    metadata.insert("source".into(), json!(source_path));
    metadata.insert("content_type".into(), json!("text"));
    Some(Document::new(parts.join("\n"), metadata))
}

fn build_table_documents(tables: &[(&Table, Option<&str>)], source_path: &str) -> Vec<Document> {
    let source_name = Path::new(source_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| source_path.to_string());

    let mut docs = Vec::new();

    for (table_index, (table, _)) in tables.iter().enumerate() {
        let table_index = table_index + 1;
        let caption = table
            .caption
            .as_deref()
            .map(str::trim)
            .filter(|c| !c.is_empty());
        let (columns, body) = split_header(table);

        for (row_index, row) in body.iter().enumerate() {
            let row_text = format_table_row_as_text(row, &columns);
            if row_text.is_empty() {
                continue;
            }

            let mut parts = vec![format!("{source_name}.")];
            if let Some(c) = caption {
                parts.push(format!("Tabela: {c}."));
            }
            parts.push(row_text);

            let mut metadata = Map::new();
            metadata.insert("source".into(), json!(source_path));
            metadata.insert("content_type".into(), json!("table_row"));
            metadata.insert("table_index".into(), json!(table_index));
            metadata.insert("table_row_index".into(), json!(row_index + 1));
            metadata.insert(
                "table_caption".into(),
                caption.map_or(Value::Null, |c| json!(c)),
            );
            docs.push(Document::new(parts.join("\n"), metadata));
        }
    }

    docs
}

/// Separa cabeçalho (colunas) e corpo. Cabeçalhos de várias linhas são
/// unidos por coluna, sem repetir valores iguais.
fn split_header(table: &Table) -> (Vec<String>, &[Vec<String>]) {
    let n_header = table.header_row_count().max(1).min(table.rows.len());
    let width = table.rows.iter().map(Vec::len).max().unwrap_or(0);

    let columns = (0..width)
        .map(|c| {
            let mut name: Vec<&str> = Vec::new();
            for row in &table.rows[..n_header] {
                let cell = row.get(c).map(|s| s.trim()).unwrap_or("");
                if !cell.is_empty() && !name.contains(&cell) {
                    name.push(cell);
                }
            }
            name.join(" ")
        })
        .collect();

    (columns, &table.rows[n_header..])
}

pub(crate) fn is_empty_value(value: &str) -> bool {
    let v = value.trim();
    v.is_empty() || v.eq_ignore_ascii_case("nan") || v.eq_ignore_ascii_case("none")
}

pub(crate) fn format_table_row_as_text(row: &[String], columns: &[String]) -> String {
    row.iter()
        .zip(columns)
        .filter(|(value, _)| !is_empty_value(value))
        .map(|(value, column)| {
            let column = column.trim();
            let column = if column.is_empty() { "Campo" } else { column };
            format!("{column}: {}.", value.trim())
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Número de páginas de um PDF, sem convertê-lo. `None` se não for um PDF legível.
pub fn pdf_page_count(path: impl AsRef<Path>) -> Option<usize> {
    let bytes = std::fs::read(path).ok()?;
    docling::pdf_page_count(&bytes, None).ok()
}

/// Carrega todos os modelos do pipeline de PDF/imagem (layout, OCR, TableFormer e o pool de
/// páginas). Efeito útil: grava o cache de grafos ONNX otimizados, para que processos que só
/// podem LER esse cache não paguem a otimização a cada documento.
pub fn warm_up_models() -> Result<(), String> {
    let mut pipeline = docling_pdf::Pipeline::new().map_err(|e| e.to_string())?;
    pipeline.warm_up_all().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn empty_values() {
        for v in ["", "  ", "nan", "NaN", "None", "none"] {
            assert!(is_empty_value(v), "{v:?}");
        }
        assert!(!is_empty_value("0"));
    }

    #[test]
    fn row_as_text_skips_empty_and_names_blank_columns() {
        let row = format_table_row_as_text(&s(&["a", "nan", "c"]), &s(&["X", "Y", ""]));
        assert_eq!(row, "X: a. Campo: c.");
    }

    #[test]
    fn table_rows_become_documents() {
        let table = Table {
            rows: vec![s(&["Nome", "Idade"]), s(&["Ana", "30"]), s(&["", "nan"])],
            caption: Some("Pessoas".into()),
            ..Default::default()
        };
        let docs = build_table_documents(&[(&table, None)], "/tmp/arq.pdf");
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].page_content,
            "arq.pdf.\nTabela: Pessoas.\nNome: Ana. Idade: 30."
        );
        assert_eq!(docs[0].metadata["table_row_index"], json!(1));
    }

    #[test]
    fn empty_text_yields_no_document() {
        assert!(build_text_document(&[], "x").is_none());
    }
}
