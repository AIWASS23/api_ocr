//! `legal` (nova) — legislação brasileira (leis, decretos, portarias, resoluções...). Cada
//! artigo vira um chunk, com seus parágrafos, incisos e alíneas, prefixado pelo caminho
//! estrutural: `LEI Nº 1.234/2020 > TÍTULO I - ... > CAPÍTULO II - ... > Seção I`. Um artigo
//! maior que o `chunk_size` é dividido nas fronteiras de `§`/`Parágrafo único`/incisos/alíneas;
//! os pedaços seguintes repetem o rótulo do artigo (`Art. 5º (continuação)`).
//!
//! Sem nenhum artigo reconhecido, cai no corte recursivo.
//!
//! Opções: `breadcrumb` (`true`), `merge_small_articles` (`false`: junta artigos curtos
//! seguidos da mesma seção até o `chunk_size`). Metadados: `norma`, `parte`, `livro`, `titulo`,
//! `capitulo`, `secao`, `subsecao`, `artigo` (os presentes).

use regex::Regex;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::{Document, Metadata};
use crate::error::Result;
use crate::strategies::recursive::{ChunkRecursive, enforce_size};

const LEVELS: [&str; 6] = ["parte", "livro", "titulo", "capitulo", "secao", "subsecao"];

pub struct ChunkLegal {
    sizing: Sizing,
    breadcrumb: bool,
    merge_small: bool,
    heading: Regex,
    article: Regex,
    norm: Regex,
    subdivision: Regex,
    fallback: ChunkRecursive,
}

#[derive(Clone, Default)]
struct Path {
    norma: Option<String>,
    levels: [Option<String>; 6],
}

impl Path {
    fn crumbs(&self) -> Vec<String> {
        self.norma.iter().cloned().chain(self.levels.iter().flatten().cloned()).collect()
    }

    fn metadata(&self) -> Metadata {
        let mut m = Metadata::new();
        if let Some(n) = &self.norma {
            m.insert("norma".into(), n.clone().into());
        }
        for (k, v) in LEVELS.iter().zip(&self.levels) {
            if let Some(v) = v {
                m.insert((*k).into(), v.clone().into());
            }
        }
        m
    }
}

struct Unit {
    path: Path,
    article: Option<String>,
    lines: Vec<String>,
}

fn level_of(word: &str) -> usize {
    let w = word.to_lowercase();
    if w.starts_with("parte") {
        0
    } else if w.starts_with("livro") {
        1
    } else if w.starts_with("t") {
        2
    } else if w.starts_with("cap") {
        3
    } else if w.starts_with("sub") {
        5
    } else {
        4
    }
}

impl ChunkLegal {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let breadcrumb = o.bool("breadcrumb", true)?;
        let merge_small = o.bool("merge_small_articles", false)?;
        o.finish()?;
        let rx = |p: &str| Regex::new(p).expect("regex fixa");
        Ok(Self {
            sizing,
            breadcrumb,
            merge_small,
            heading: rx(r"(?i)^\s*(PARTE|LIVRO|T[ÍI]TULO|CAP[ÍI]TULO|SUBSE[ÇC][ÃA]O|SE[ÇC][ÃA]O)\s+([IVXLCDM]+|\d+|[ÚU]NIC[AO])\b\s*[-–—.:]?\s*(.*)$"),
            article: rx(r"^\s*(?:Art\.?|Artigo)\s*(\d+(?:\.\d+)*)\s*(º|°|ª|o\b)?(?:\s*[-–]\s*([A-Z])\b)?"),
            norm: rx(r"(?i)^\s*(LEI(?:\s+COMPLEMENTAR|\s+ORG[ÂA]NICA|\s+ORDIN[ÁA]RIA)?|DECRETO(?:-LEI|\s+LEGISLATIVO)?|PORTARIA|RESOLU[ÇC][ÃA]O|INSTRU[ÇC][ÃA]O\s+NORMATIVA|EMENDA\s+CONSTITUCIONAL|MEDIDA\s+PROVIS[ÓO]RIA|DELIBERA[ÇC][ÃA]O)\b"),
            subdivision: rx(r"^\s*(?:§\s*\d+|Par[áa]grafo\s+[úu]nico|[IVXLCDM]+\s*[-–—]|[a-z]\))"),
            fallback: ChunkRecursive::new(sizing)?,
        })
    }

    /// Nome de divisão na linha seguinte ("DAS DISPOSIÇÕES GERAIS", "Do Lançamento"): curto,
    /// com letras e sem pontuação final de frase.
    fn is_division_name(line: &str) -> bool {
        let t = strip(line);
        !t.is_empty() && pystr::len(t) <= 120 && t.chars().any(char::is_alphabetic) && !t.ends_with(['.', ';', ':', ','])
    }

    fn units(&self, text: &str) -> Vec<Unit> {
        let lines: Vec<&str> = text.split('\n').collect();
        let mut path = Path::default();
        let mut units: Vec<Unit> = vec![Unit { path: Path::default(), article: None, lines: Vec::new() }];
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            if path.norma.is_none() && units.len() == 1 && self.norm.is_match(line) && pystr::len(strip(line)) <= 200 {
                path.norma = Some(strip(line).to_string());
                units[0].path = path.clone();
                i += 1;
                continue;
            }
            if let Some(c) = self.heading.captures(line)
                && !self.article.is_match(line)
            {
                let level = level_of(&c[1]);
                let mut label = format!("{} {}", strip(&c[1]), &c[2]);
                let rest = strip(c.get(3).map_or("", |m| m.as_str()));
                if !rest.is_empty() {
                    label = format!("{label} - {rest}");
                } else if let Some(next) = lines.get(i + 1).copied()
                    && Self::is_division_name(next)
                    && !self.article.is_match(next)
                    && !self.heading.is_match(next)
                {
                    label = format!("{label} - {}", strip(next));
                    i += 1;
                }
                path.levels[level] = Some(label);
                for l in path.levels.iter_mut().skip(level + 1) {
                    *l = None;
                }
                units.push(Unit { path: path.clone(), article: None, lines: Vec::new() });
                i += 1;
                continue;
            }
            if let Some(c) = self.article.captures(line) {
                let mut label = format!("Art. {}", &c[1]);
                if let Some(o) = c.get(2) {
                    label.push_str(if o.as_str() == "o" { "º" } else { o.as_str() });
                }
                if let Some(l) = c.get(3) {
                    label.push('-');
                    label.push_str(l.as_str());
                }
                units.push(Unit { path: path.clone(), article: Some(label), lines: vec![line.to_string()] });
                i += 1;
                continue;
            }
            units.last_mut().expect("sempre há uma unidade").lines.push(line.to_string());
            i += 1;
        }
        units.retain(|u| !strip(&u.lines.join("\n")).is_empty());
        units
    }

    /// Partes de um artigo nas fronteiras de `§`, incisos e alíneas.
    fn parts(&self, lines: &[String]) -> Vec<String> {
        let mut parts: Vec<Vec<&str>> = Vec::new();
        for l in lines {
            if parts.is_empty() || self.subdivision.is_match(l) {
                parts.push(vec![l.as_str()]);
            } else {
                parts.last_mut().expect("não vazio").push(l.as_str());
            }
        }
        parts.into_iter().map(|p| p.join("\n")).collect()
    }

    fn render(&self, unit: &Unit) -> Result<Vec<Document>> {
        let crumbs = unit.path.crumbs();
        let prefix = if self.breadcrumb && !crumbs.is_empty() { crumbs.join(" > ") } else { String::new() };
        let body = strip(&unit.lines.join("\n")).to_string();
        let mut meta = unit.path.metadata();
        if let Some(a) = &unit.article {
            meta.insert("artigo".into(), a.clone().into());
        }
        let wrap = |p: &str, b: &str| if p.is_empty() { b.to_string() } else { format!("{p}\n{b}") };
        let whole = wrap(&prefix, &body);
        if pystr::len(&whole) <= self.sizing.chunk_size {
            return Ok(vec![Document::with_metadata(whole, meta)]);
        }
        let cont = match &unit.article {
            Some(a) => wrap(&prefix, &format!("{a} (continuação)")),
            None => prefix.clone(),
        };
        let room = |p: &str| self.sizing.chunk_size.saturating_sub(pystr::len(p) + 1).max(1);
        let mut out: Vec<String> = Vec::new();
        let mut current = String::new();
        for part in self.parts(&unit.lines) {
            let head = if out.is_empty() && current.is_empty() { prefix.as_str() } else { cont.as_str() };
            let candidate = if current.is_empty() { part.clone() } else { format!("{current}\n{part}") };
            if pystr::len(&candidate) <= room(head) {
                current = candidate;
                continue;
            }
            if !current.is_empty() {
                let h = if out.is_empty() { prefix.as_str() } else { cont.as_str() };
                out.push(wrap(h, &current));
                current.clear();
            }
            if pystr::len(&part) <= room(&cont) {
                current = part;
            } else {
                let r = room(&cont);
                let sz = Sizing::new(r, self.sizing.chunk_overlap.min(r - 1))?;
                for piece in enforce_size(&[part], sz)? {
                    let h = if out.is_empty() { prefix.as_str() } else { cont.as_str() };
                    out.push(wrap(h, &piece));
                }
            }
        }
        if !current.is_empty() {
            let h = if out.is_empty() { prefix.as_str() } else { cont.as_str() };
            out.push(wrap(h, &current));
        }
        Ok(out.into_iter().map(|c| Document::with_metadata(c, meta.clone())).collect())
    }
}

impl Chunker for ChunkLegal {
    crate::sizing_accessors!("legal");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.chunk_documents(text)?.into_iter().map(|d| d.page_content).collect())
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let units = self.units(text);
        if !units.iter().any(|u| u.article.is_some()) {
            return Ok(self.fallback.chunk(text)?.into_iter().map(Document::new).collect());
        }
        let mut docs: Vec<Document> = Vec::new();
        for u in &units {
            docs.extend(self.render(u)?);
        }
        if !self.merge_small {
            return Ok(docs);
        }
        // Junta artigos seguidos do mesmo caminho estrutural enquanto couberem.
        let mut merged: Vec<Document> = Vec::new();
        for d in docs {
            if let Some(last) = merged.last_mut() {
                let same = LEVELS.iter().chain(["norma"].iter()).all(|k| last.metadata.get(*k) == d.metadata.get(*k));
                let body = d.page_content.split_once('\n').map_or(d.page_content.as_str(), |(_, b)| b);
                let joined = format!("{}\n\n{}", last.page_content, body);
                if same && pystr::len(&joined) <= self.sizing.chunk_size {
                    let arts = format!(
                        "{}, {}",
                        last.metadata.get("artigo").and_then(|v| v.as_str()).unwrap_or(""),
                        d.metadata.get("artigo").and_then(|v| v.as_str()).unwrap_or("")
                    );
                    last.page_content = joined;
                    last.metadata.insert("artigo".into(), arts.trim_matches([',', ' ']).to_string().into());
                    continue;
                }
            }
            merged.push(d);
        }
        Ok(merged)
    }
}
