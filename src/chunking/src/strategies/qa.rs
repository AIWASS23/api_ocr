//! `qa` (nova) — FAQ e documentos de perguntas e respostas: cada par vira um chunk
//! (`Pergunta: ...\nResposta: ...`), então a busca acha a resposta pela pergunta do usuário.
//! Reconhece, nesta ordem: rótulos explícitos (`Pergunta:`/`P:`/`Q:` e `Resposta:`/`R:`/`A:`),
//! títulos Markdown terminados em `?`, linhas em negrito terminadas em `?` e linhas curtas
//! terminadas em `?`. Resposta longa é dividida e cada pedaço repete a pergunta. O texto antes
//! da primeira pergunta é cortado pelo splitter recursivo; sem perguntas, tudo é.
//!
//! Opções: `format` (`"labeled"` | `"original"`), `max_question_chars` (300).
//! Metadados: `question`, `qa_index`.

use regex::Regex;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pystr::{self, strip};
use crate::document::Document;
use crate::error::{ChunkError, Result};
use crate::strategies::recursive::{ChunkRecursive, enforce_size};

pub struct ChunkQa {
    sizing: Sizing,
    labeled: bool,
    max_question: usize,
    q_label: Regex,
    a_label: Regex,
    fallback: ChunkRecursive,
}

struct Pair {
    question: String,
    answer: Vec<String>,
}

impl ChunkQa {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let labeled = match o.string("format")?.as_deref() {
            None | Some("labeled") => true,
            Some("original") => false,
            Some(other) => return Err(ChunkError::config(format!("qa: format '{other}' (use labeled ou original)"))),
        };
        let max_question = o.usize("max_question_chars")?.unwrap_or(300);
        o.finish()?;
        let rx = |p: &str| Regex::new(p).expect("regex fixa");
        Ok(Self {
            sizing,
            labeled,
            max_question,
            q_label: rx(r"(?i)^\s*(?:\*\*)?(?:pergunta|quest[ãa]o|d[úu]vida|p|q)\s*\d*\s*(?:\*\*)?\s*[:\-–]\s*(?:\*\*)?\s*(.+)$"),
            a_label: rx(r"(?i)^\s*(?:\*\*)?(?:resposta|r|a)\s*(?:\*\*)?\s*[:\-–]\s*(?:\*\*)?\s*(.*)$"),
            fallback: ChunkRecursive::new(sizing)?,
        })
    }

    /// Pergunta implícita: título `# ...?`, negrito `**...?**` ou linha curta terminada em `?`.
    fn implicit_question(&self, line: &str) -> Option<String> {
        let t = strip(line);
        let t = t.trim_start_matches('#').trim();
        let t = t.strip_prefix("**").and_then(|x| x.strip_suffix("**")).unwrap_or(t).trim();
        let t = t.trim_start_matches(|c: char| c.is_ascii_digit() || matches!(c, '.' | ')' | '-' | ' '));
        (t.ends_with('?') && pystr::len(t) <= self.max_question && pystr::len(t) >= 5).then(|| t.to_string())
    }

    fn pairs(&self, text: &str) -> (Vec<String>, Vec<Pair>) {
        let lines: Vec<&str> = text.split('\n').collect();
        let explicit = lines.iter().any(|l| self.q_label.is_match(l));
        let mut preamble = Vec::new();
        let mut pairs: Vec<Pair> = Vec::new();
        for line in lines {
            let question = if explicit {
                self.q_label.captures(line).map(|c| strip(&c[1]).to_string())
            } else {
                self.implicit_question(line)
            };
            match (question, pairs.last_mut()) {
                (Some(q), _) => pairs.push(Pair { question: q, answer: Vec::new() }),
                (None, Some(p)) => {
                    let ans = if explicit {
                        self.a_label.captures(line).map_or(line.to_string(), |c| c[1].to_string())
                    } else {
                        line.to_string()
                    };
                    p.answer.push(ans);
                }
                (None, None) => preamble.push(line.to_string()),
            }
        }
        (preamble, pairs)
    }

    fn render(&self, q: &str, a: &str) -> String {
        if self.labeled { format!("Pergunta: {q}\nResposta: {a}") } else { format!("{q}\n{a}") }
    }
}

impl Chunker for ChunkQa {
    crate::sizing_accessors!("qa");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.chunk_documents(text)?.into_iter().map(|d| d.page_content).collect())
    }

    fn chunk_documents(&self, text: &str) -> Result<Vec<Document>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let (preamble, pairs) = self.pairs(text);
        if pairs.is_empty() {
            return Ok(self.fallback.chunk(text)?.into_iter().map(Document::new).collect());
        }
        let mut out: Vec<Document> = self.fallback.chunk(&preamble.join("\n"))?.into_iter().map(Document::new).collect();
        for (i, p) in pairs.iter().enumerate() {
            let answer = strip(&p.answer.join("\n")).to_string();
            let whole = self.render(&p.question, &answer);
            let meta = |d: Document| d.insert("question", p.question.as_str()).insert("qa_index", i as u64);
            if pystr::len(&whole) <= self.sizing.chunk_size || answer.is_empty() {
                out.push(meta(Document::new(whole)));
                continue;
            }
            let overhead = pystr::len(&self.render(&p.question, ""));
            let room = self.sizing.chunk_size.saturating_sub(overhead).max(1);
            let sz = Sizing::new(room, self.sizing.chunk_overlap.min(room - 1))?;
            for piece in enforce_size(&[answer], sz)? {
                out.push(meta(Document::new(self.render(&p.question, &piece))));
            }
        }
        Ok(out)
    }
}
