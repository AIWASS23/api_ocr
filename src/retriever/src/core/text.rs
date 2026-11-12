//! Tokens (`re.findall(r"\w+", texto.lower())` do Python) e a razão de semelhança do
//! `difflib.SequenceMatcher` (Ratcliff/Obershelp, com o `autojunk` do Python).

use std::collections::HashMap;

use embedding::core::analyzer::is_word_char;

/// Sequências de caracteres de palavra (`\w` do Python), em minúsculas.
pub fn tokens(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if is_word_char(c) {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

struct Matcher<'a> {
    a: &'a [char],
    b: &'a [char],
    b2j: HashMap<char, Vec<usize>>,
}

impl<'a> Matcher<'a> {
    fn new(a: &'a [char], b: &'a [char]) -> Self {
        let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
        for (j, &c) in b.iter().enumerate() {
            b2j.entry(c).or_default().push(j);
        }
        // autojunk: em `b` com 200+ elementos, os que aparecem em mais de 1% saem do índice.
        let n = b.len();
        if n >= 200 {
            let ntest = n / 100 + 1;
            b2j.retain(|_, idx| idx.len() <= ntest);
        }
        Self { a, b, b2j }
    }

    /// `find_longest_match` (sem junk explícito: só os populares saem do índice).
    fn longest(&self, alo: usize, ahi: usize, blo: usize, bhi: usize) -> (usize, usize, usize) {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut new: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = self.b2j.get(&self.a[i]) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = j2len.get(&(j.wrapping_sub(1))).copied().filter(|_| j > 0).unwrap_or(0) + 1;
                    new.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = new;
        }
        while besti > alo && bestj > blo && self.a[besti - 1] == self.b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi && bestj + bestsize < bhi && self.a[besti + bestsize] == self.b[bestj + bestsize] {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    }

    fn matches(&self) -> usize {
        let mut queue = vec![(0, self.a.len(), 0, self.b.len())];
        let mut total = 0;
        while let Some((alo, ahi, blo, bhi)) = queue.pop() {
            let (i, j, k) = self.longest(alo, ahi, blo, bhi);
            if k > 0 {
                total += k;
                if alo < i && blo < j {
                    queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
        total
    }
}

/// `SequenceMatcher(None, a, b).ratio()`.
pub fn ratio(a: &str, b: &str) -> f64 {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let total = a.len() + b.len();
    if total == 0 {
        return 1.0;
    }
    2.0 * Matcher::new(&a, &b).matches() as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_like_python() {
        assert_eq!(tokens("Olá, MUNDO! a_b 42"), vec!["olá", "mundo", "a_b", "42"]);
    }

    #[test]
    fn difflib_ratio() {
        // difflib.SequenceMatcher(None, a, b).ratio()
        assert!((ratio("abcd", "bcde") - 0.75).abs() < 1e-12);
        assert_eq!(ratio("", ""), 1.0);
        assert!((ratio("python", "pythonic") - 0.857_142_857_142_857_1).abs() < 1e-12);
        assert_eq!(ratio("abc", "xyz"), 0.0);
    }
}
