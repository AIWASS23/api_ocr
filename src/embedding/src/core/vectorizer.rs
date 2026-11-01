//! `CountVectorizer` e `TfidfTransformer` do scikit-learn: vocabulário aprendido nos documentos
//! (ordem alfabética, poda por `max_df`/`min_df`/`max_features`) e pesos TF-IDF. Base de `tfidf`,
//! `bm25` e `lsa`. Matrizes esparsas por linha, em `f64` como no Python.

use std::collections::HashMap;

use serde_json::Value;

use crate::core::analyzer::Analyzer;
use crate::core::options::Options;
use crate::error::{EmbeddingError, Result};

/// Linha esparsa: `(coluna, valor)` em ordem de coluna.
pub type SparseRow = Vec<(usize, f64)>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DocFreq {
    Count(usize),
    Fraction(f64),
}

impl DocFreq {
    fn parse(o: &mut Options, key: &str, default: DocFreq) -> Result<DocFreq> {
        match o.raw(key) {
            None => Ok(default),
            Some(Value::Number(n)) if n.is_u64() => {
                let c = n.as_u64().unwrap_or(0) as usize;
                if c < 1 {
                    return Err(EmbeddingError::config(format!("{key} must be a float in [0.0, 1.0] or an int >= 1")));
                }
                Ok(DocFreq::Count(c))
            }
            Some(Value::Number(n)) if n.as_f64().is_some_and(|f| (0.0..=1.0).contains(&f)) => Ok(DocFreq::Fraction(n.as_f64().unwrap_or(1.0))),
            Some(_) => Err(EmbeddingError::config(format!("{key} must be a float in [0.0, 1.0] or an int >= 1"))),
        }
    }

    fn resolve(self, n_doc: usize) -> f64 {
        match self {
            DocFreq::Count(c) => c as f64,
            DocFreq::Fraction(f) => f * n_doc as f64,
        }
    }
}

/// Vocabulário: termo de cada coluna e o índice de cada termo.
#[derive(Clone, Debug, Default)]
pub struct Vocabulary {
    pub terms: Vec<String>,
    pub index: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    fn from_terms(terms: Vec<String>) -> Self {
        let index = terms.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        Self { terms, index }
    }
}

pub struct CountVectorizer {
    pub analyzer: Analyzer,
    max_df: DocFreq,
    min_df: DocFreq,
    max_features: Option<usize>,
    binary: bool,
    fixed: Option<Vocabulary>,
}

impl CountVectorizer {
    /// Opções do `CountVectorizer` (as do analisador, `max_df`, `min_df`, `max_features`,
    /// `vocabulary`, `binary`). `default_max_features`: o `dimensions or 4096` do Python.
    pub fn from_options(o: &mut Options, default_max_features: Option<usize>) -> Result<Self> {
        let analyzer = Analyzer::from_options(o)?;
        let max_df = DocFreq::parse(o, "max_df", DocFreq::Fraction(1.0))?;
        let min_df = DocFreq::parse(o, "min_df", DocFreq::Count(1))?;
        let max_features = match o.raw("max_features") {
            None => default_max_features,
            Some(v) => Some(v.as_u64().filter(|&n| n >= 1).ok_or_else(|| EmbeddingError::config("max_features must be a positive integer or None"))? as usize),
        };
        let binary = o.bool("binary", false)?;
        let fixed = match o.raw("vocabulary") {
            None => None,
            Some(Value::Array(a)) => {
                let terms: Vec<String> = a
                    .into_iter()
                    .map(|v| v.as_str().map(str::to_string).ok_or_else(|| EmbeddingError::config("vocabulary must contain strings")))
                    .collect::<Result<_>>()?;
                let vocab = Vocabulary::from_terms(terms);
                if vocab.index.len() != vocab.terms.len() {
                    return Err(EmbeddingError::config("Duplicate term in vocabulary"));
                }
                Some(vocab)
            }
            Some(Value::Object(m)) => {
                let mut terms = vec![String::new(); m.len()];
                let mut seen = vec![false; m.len()];
                for (term, idx) in m {
                    let i = idx.as_u64().map(|i| i as usize).filter(|&i| i < seen.len() && !seen[i]).ok_or_else(|| {
                        EmbeddingError::config("vocabulary indices must be a permutation of 0..len(vocabulary)")
                    })?;
                    seen[i] = true;
                    terms[i] = term;
                }
                Some(Vocabulary::from_terms(terms))
            }
            Some(_) => return Err(EmbeddingError::config("vocabulary must be a list of terms or a {term: index} mapping")),
        };
        if fixed.as_ref().is_some_and(Vocabulary::is_empty) {
            return Err(EmbeddingError::config("empty vocabulary passed to fit"));
        }
        Ok(Self { analyzer, max_df, min_df, max_features, binary, fixed })
    }

    fn count_doc(&self, doc: &str, vocab: &Vocabulary) -> SparseRow {
        let mut counts: HashMap<usize, f64> = HashMap::new();
        for term in self.analyzer.analyze(doc) {
            if let Some(&i) = vocab.index.get(&term) {
                *counts.entry(i).or_insert(0.0) += 1.0;
            }
        }
        let mut row: SparseRow = counts.into_iter().map(|(i, c)| (i, if self.binary { 1.0 } else { c })).collect();
        row.sort_unstable_by_key(|&(i, _)| i);
        row
    }

    /// Aprende o vocabulário e devolve a matriz de contagens (`fit_transform`).
    pub fn fit_transform(&self, docs: &[&str]) -> Result<(Vocabulary, Vec<SparseRow>)> {
        if let Some(fixed) = &self.fixed {
            return Ok((fixed.clone(), docs.iter().map(|d| self.count_doc(d, fixed)).collect()));
        }
        // Contagem com o vocabulário na ordem de aparição, como o `_count_vocab`.
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut terms: Vec<String> = Vec::new();
        let mut rows: Vec<HashMap<usize, f64>> = Vec::with_capacity(docs.len());
        for doc in docs {
            let mut counts: HashMap<usize, f64> = HashMap::new();
            for term in self.analyzer.analyze(doc) {
                let i = *index.entry(term.clone()).or_insert_with(|| {
                    terms.push(term);
                    terms.len() - 1
                });
                *counts.entry(i).or_insert(0.0) += 1.0;
            }
            rows.push(counts);
        }
        if terms.is_empty() {
            return Err(EmbeddingError::input("empty vocabulary; perhaps the documents only contain stop words"));
        }
        if self.binary {
            for r in rows.iter_mut() {
                r.values_mut().for_each(|v| *v = 1.0);
            }
        }
        let n_doc = docs.len();
        let (high, low) = (self.max_df.resolve(n_doc), self.min_df.resolve(n_doc));
        if high < low {
            return Err(EmbeddingError::config("max_df corresponds to < documents than min_df"));
        }
        // Colunas em ordem alfabética (`_sort_features`).
        let mut order: Vec<usize> = (0..terms.len()).collect();
        order.sort_by(|&a, &b| terms[a].cmp(&terms[b]));
        let mut df = vec![0usize; terms.len()];
        let mut tf = vec![0f64; terms.len()];
        for r in &rows {
            for (&i, &v) in r {
                df[i] += 1;
                tf[i] += v;
            }
        }
        // `_limit_features`: frequência de documento e, se preciso, os `max_features` mais
        // frequentes (empate: ordem alfabética).
        let mut kept: Vec<usize> = order.iter().copied().filter(|&i| (df[i] as f64) <= high && (df[i] as f64) >= low).collect();
        if let Some(limit) = self.max_features
            && kept.len() > limit
        {
            let mut by_tf = kept.clone();
            by_tf.sort_by(|&a, &b| tf[b].partial_cmp(&tf[a]).unwrap_or(std::cmp::Ordering::Equal));
            by_tf.truncate(limit);
            let chosen: std::collections::HashSet<usize> = by_tf.into_iter().collect();
            kept.retain(|i| chosen.contains(i));
        }
        if kept.is_empty() {
            return Err(EmbeddingError::input("After pruning, no terms remain. Try a lower min_df or a higher max_df."));
        }
        let mut new_index = vec![usize::MAX; terms.len()];
        for (new, &old) in kept.iter().enumerate() {
            new_index[old] = new;
        }
        let vocab = Vocabulary::from_terms(kept.iter().map(|&i| terms[i].clone()).collect());
        let matrix = rows
            .into_iter()
            .map(|r| {
                let mut row: SparseRow = r.into_iter().filter_map(|(i, v)| (new_index[i] != usize::MAX).then_some((new_index[i], v))).collect();
                row.sort_unstable_by_key(|&(i, _)| i);
                row
            })
            .collect();
        Ok((vocab, matrix))
    }

    /// Contagens com um vocabulário já aprendido (termos novos são ignorados).
    pub fn transform(&self, vocab: &Vocabulary, docs: &[&str]) -> Vec<SparseRow> {
        docs.iter().map(|d| self.count_doc(d, vocab)).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    L1,
    L2,
}

pub fn parse_norm(o: &mut Options, default: Option<Norm>) -> Result<Option<Norm>> {
    if !o.has("norm") {
        return Ok(default);
    }
    match o.raw("norm") {
        None => Ok(None),
        Some(Value::String(s)) if s == "l2" => Ok(Some(Norm::L2)),
        Some(Value::String(s)) if s == "l1" => Ok(Some(Norm::L1)),
        Some(_) => Err(EmbeddingError::config("norm must be 'l1', 'l2' or None")),
    }
}

/// `sklearn.preprocessing.normalize` por linha (linhas nulas ficam como estão).
pub fn normalize_rows(rows: &mut [SparseRow], norm: Norm) {
    for row in rows.iter_mut() {
        let n = match norm {
            Norm::L1 => row.iter().map(|(_, v)| v.abs()).sum::<f64>(),
            Norm::L2 => row.iter().map(|(_, v)| v * v).sum::<f64>().sqrt(),
        };
        if n != 0.0 {
            row.iter_mut().for_each(|(_, v)| *v /= n);
        }
    }
}

/// `TfidfTransformer`.
#[derive(Clone, Debug)]
pub struct TfidfWeights {
    pub norm: Option<Norm>,
    pub use_idf: bool,
    pub smooth_idf: bool,
    pub sublinear_tf: bool,
    pub idf: Vec<f64>,
}

impl TfidfWeights {
    pub fn from_options(o: &mut Options) -> Result<Self> {
        Ok(Self {
            norm: parse_norm(o, Some(Norm::L2))?,
            use_idf: o.bool("use_idf", true)?,
            smooth_idf: o.bool("smooth_idf", true)?,
            sublinear_tf: o.bool("sublinear_tf", false)?,
            idf: Vec::new(),
        })
    }

    /// `idf = ln((n + s) / (df + s)) + 1`, com `s = 1` quando `smooth_idf`.
    pub fn fit(&mut self, counts: &[SparseRow], n_features: usize) {
        let mut df = vec![0f64; n_features];
        for row in counts {
            for &(i, v) in row {
                if v != 0.0 {
                    df[i] += 1.0;
                }
            }
        }
        let s = if self.smooth_idf { 1.0 } else { 0.0 };
        let n = counts.len() as f64 + s;
        self.idf = df.iter().map(|&d| (n / (d + s)).ln() + 1.0).collect();
    }

    pub fn transform(&self, mut rows: Vec<SparseRow>) -> Vec<SparseRow> {
        for row in rows.iter_mut() {
            for (i, v) in row.iter_mut() {
                if self.sublinear_tf {
                    *v = v.ln() + 1.0;
                }
                if self.use_idf {
                    *v *= self.idf[*i];
                }
            }
        }
        if let Some(norm) = self.norm {
            normalize_rows(&mut rows, norm);
        }
        rows
    }
}

/// Linha esparsa em `f64` → vetor denso `f32` de largura `width` (o `toarray()` + `float32`).
pub fn densify(row: &SparseRow, width: usize) -> Vec<f32> {
    let mut out = vec![0f32; width];
    for &(i, v) in row {
        out[i] = v as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vectorizer(options: Value, default_max: Option<usize>) -> CountVectorizer {
        let map = options.as_object().cloned().unwrap_or_default();
        let mut o = Options::new("tfidf", &map);
        let v = CountVectorizer::from_options(&mut o, default_max).unwrap();
        o.finish().unwrap();
        v
    }

    #[test]
    fn vocabulary_is_sorted_and_pruned() {
        let v = vectorizer(json!({"min_df": 2}), None);
        let (vocab, rows) = v.fit_transform(&["bb aa cc aa", "aa bb", "dd"]).unwrap();
        assert_eq!(vocab.terms, ["aa", "bb"]);
        assert_eq!(rows, vec![vec![(0, 2.0), (1, 1.0)], vec![(0, 1.0), (1, 1.0)], vec![]]);
        // max_features: os mais frequentes, colunas ainda em ordem alfabética.
        let v = vectorizer(json!({}), Some(2));
        let (vocab, _) = v.fit_transform(&["zz zz yy", "zz xx yy"]).unwrap();
        assert_eq!(vocab.terms, ["yy", "zz"]);
        assert!(vectorizer(json!({"stop_words": ["aa"]}), None).fit_transform(&["aa", "aa a"]).is_err());
    }

    #[test]
    fn tfidf_like_sklearn() {
        let v = vectorizer(json!({}), None);
        let (vocab, counts) = v.fit_transform(&["aa bb", "aa"]).unwrap();
        let mut w = TfidfWeights { norm: Some(Norm::L2), use_idf: true, smooth_idf: true, sublinear_tf: false, idf: vec![] };
        w.fit(&counts, vocab.len());
        // idf = ln(3/3)+1 = 1 (aa) e ln(3/2)+1 (bb).
        assert!((w.idf[1] - ((1.5f64).ln() + 1.0)).abs() < 1e-12);
        let rows = w.transform(counts);
        let n: f64 = rows[0].iter().map(|(_, v)| v * v).sum();
        assert!((n - 1.0).abs() < 1e-12);
        assert_eq!(rows[1], vec![(0, 1.0)]);
    }
}
