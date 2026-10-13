//! `table` (nova) — mantém tabelas Markdown inteiras. Uma tabela que cabe no `chunk_size` vira
//! um chunk só; uma maior é dividida em grupos de linhas, cada grupo repetindo o cabeçalho, para
//! que nenhum valor fique sem o nome da coluna. A legenda (linha curta logo acima da tabela)
//! acompanha cada pedaço. O texto fora das tabelas é cortado pelo splitter recursivo.
//!
//! O corte por tamanho das outras estratégias parte tabelas no meio de uma linha, e o pedaço
//! seguinte perde o cabeçalho; é o problema que esta estratégia resolve.
//!
//! Opções: `format` (`"markdown"` | `"key_value"`: cada linha vira `Coluna: valor; ...`),
//! `caption` (`true`). Metadados: `kind` (`table`/`text`), `table_index`, `rows` (`"1-12"`),
//! `caption`.

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::Document;
use crate::error::{ChunkError, Result};
use crate::strategies::recursive::{ChunkRecursive, enforce_size};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Format {
    Markdown,
    KeyValue,
}

pub struct ChunkTable {
    sizing: Sizing,
    format: Format,
    caption: bool,
    text: ChunkRecursive,
}

enum Block {
    Text(String),
    Table { caption: Option<String>, header: String, separator: String, rows: Vec<String> },
}

fn is_separator_row(line: &str) -> bool {
    let t = line.trim();
    let inner = t.trim_matches('|');
    !inner.is_empty()
        && t.contains('-')
        && inner.split('|').all(|cell| {
            let c = cell.trim();
            let c = c.trim_start_matches(':').trim_end_matches(':');
            c.len() >= 3 && c.chars().all(|ch| ch == '-')
        })
}

fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.contains('|') && !t.is_empty()
}

fn cells(row: &str) -> Vec<String> {
    let t = row.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    t.split('|').map(|c| c.trim().to_string()).collect()
}

fn parse_blocks(text: &str, take_caption: bool) -> Vec<Block> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut blocks = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if i + 1 < lines.len() && is_table_row(lines[i]) && is_separator_row(lines[i + 1]) {
            let mut caption = None;
            if take_caption {
                // Linha curta não vazia até 2 linhas acima da tabela.
                let mut back = buf.len();
                let mut blanks = 0;
                while back > 0 && strip(buf[back - 1]).is_empty() && blanks < 1 {
                    back -= 1;
                    blanks += 1;
                }
                if back > 0 {
                    let cand = strip(buf[back - 1]);
                    if !cand.is_empty() && pystr::len(cand) <= 200 && !cand.contains('|') {
                        caption = Some(cand.to_string());
                        buf.truncate(back - 1);
                    }
                }
            }
            if !buf.is_empty() {
                blocks.push(Block::Text(buf.join("\n")));
                buf.clear();
            }
            let header = lines[i].trim().to_string();
            let separator = lines[i + 1].trim().to_string();
            i += 2;
            let mut rows = Vec::new();
            while i < lines.len() && is_table_row(lines[i]) {
                rows.push(lines[i].trim().to_string());
                i += 1;
            }
            blocks.push(Block::Table { caption, header, separator, rows });
            continue;
        }
        buf.push(lines[i]);
        i += 1;
    }
    if !buf.is_empty() {
        blocks.push(Block::Text(buf.join("\n")));
    }
    blocks
}

impl ChunkTable {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let format = match o.string("format")?.as_deref() {
            None | Some("markdown") => Format::Markdown,
            Some("key_value") => Format::KeyValue,
            Some(other) => return Err(ChunkError::config(format!("table: format '{other}' (use markdown ou key_value)"))),
        };
        let caption = o.bool("caption", true)?;
        o.finish()?;
        Ok(Self { sizing, format, caption, text: ChunkRecursive::new(sizing)? })
    }

    fn render_rows(&self, header: &str, separator: &str, rows: &[String]) -> Vec<String> {
        match self.format {
            Format::Markdown => rows.to_vec(),
            Format::KeyValue => {
                let cols = cells(header);
                let _ = separator;
                rows.iter()
                    .map(|r| {
                        cells(r)
                            .iter()
                            .enumerate()
                            .filter(|(_, v)| !v.is_empty())
                            .map(|(k, v)| match cols.get(k).filter(|c| !c.is_empty()) {
                                Some(c) => format!("{c}: {v}"),
                                None => v.clone(),
                            })
                            .collect::<Vec<_>>()
                            .join("; ")
                    })
                    .collect()
            }
        }
    }

    fn table_chunks(&self, index: usize, caption: &Option<String>, header: &str, separator: &str, rows: &[String]) -> Result<Vec<Document>> {
        let head = match self.format {
            Format::Markdown => format!("{header}\n{separator}"),
            Format::KeyValue => String::new(),
        };
        let mut prefix_parts: Vec<&str> = Vec::new();
        if let Some(c) = caption {
            prefix_parts.push(c);
        }
        if !head.is_empty() {
            prefix_parts.push(&head);
        }
        let prefix = prefix_parts.join("\n");
        let rendered = self.render_rows(header, separator, rows);
        let make = |from: usize, to: usize, body: &str| {
            let content = if prefix.is_empty() { body.to_string() } else if body.is_empty() { prefix.clone() } else { format!("{prefix}\n{body}") };
            let mut d = Document::new(content)
                .insert("kind", "table")
                .insert("table_index", index as u64)
                .insert("rows", format!("{}-{}", from + 1, to));
            if let Some(c) = caption {
                d = d.insert("caption", c.as_str());
            }
            d
        };
        let fits = |s: &str| pystr::len(s) <= self.sizing.chunk_size;
        let mut out = Vec::new();
        let mut group: Vec<&str> = Vec::new();
        let mut start = 0;
        let budget = |g: &[&str], extra: &str| {
            let mut v: Vec<&str> = g.to_vec();
            v.push(extra);
            let body = v.join("\n");
            if prefix.is_empty() { body } else { format!("{prefix}\n{body}") }
        };
        for (k, row) in rendered.iter().enumerate() {
            if !group.is_empty() && !fits(&budget(&group, row)) {
                out.push(make(start, k, &group.join("\n")));
                group.clear();
                start = k;
            }
            if group.is_empty() && !fits(&budget(&[], row)) {
                // Uma linha sozinha não cabe: corta a linha, mantendo o cabeçalho em cada pedaço.
                let room = self.sizing.chunk_size.saturating_sub(pystr::len(&prefix) + 1).max(1);
                let sz = Sizing::new(room, self.sizing.chunk_overlap.min(room - 1))?;
                for piece in enforce_size(std::slice::from_ref(row), sz)? {
                    out.push(make(k, k + 1, &piece));
                }
                start = k + 1;
                continue;
            }
            group.push(row);
        }
        if !group.is_empty() || rendered.is_empty() {
            out.push(make(start, rendered.len(), &group.join("\n")));
        }
        Ok(out)
    }
}

impl Chunker for ChunkTable {
    crate::sizing_accessors!("table");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.chunk_documents(text)?.into_iter().map(|d| d.page_content).collect())
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let mut tables = 0;
        for block in parse_blocks(text, self.caption) {
            match block {
                Block::Text(t) => {
                    out.extend(self.text.chunk(&t)?.into_iter().map(|c| Document::new(c).insert("kind", "text")));
                }
                Block::Table { caption, header, separator, rows } => {
                    out.extend(self.table_chunks(tables, &caption, &header, &separator, &rows)?);
                    tables += 1;
                }
            }
        }
        Ok(out)
    }
}
