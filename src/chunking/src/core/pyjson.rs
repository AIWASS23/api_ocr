//! JSON com a semântica do módulo `json` do Python, para o `RecursiveJsonSplitter`:
//! ordem das chaves preservada, chave repetida mantém a posição da primeira ocorrência,
//! inteiros de qualquer tamanho, `NaN`/`Infinity`, e `json.dumps` com `", "`/`": "`,
//! `ensure_ascii` e floats no formato do `repr`.

use indexmap::IndexMap;

use crate::error::{ChunkError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum PyJson {
    Null,
    Bool(bool),
    /// Inteiro em texto decimal (sem limite de tamanho).
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<PyJson>),
    Dict(IndexMap<String, PyJson>),
}

struct Parser<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> ChunkError {
        ChunkError::Json(format!("{msg} (posição {})", self.i))
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn lit(&mut self, word: &str) -> bool {
        if self.src[self.i..].starts_with(word) {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<PyJson> {
        self.ws();
        let Some(&c) = self.s.get(self.i) else { return Err(self.err("Expecting value")) };
        match c {
            b'{' => {
                self.i += 1;
                let mut map = IndexMap::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(PyJson::Dict(map));
                }
                loop {
                    self.ws();
                    if self.s.get(self.i) != Some(&b'"') {
                        return Err(self.err("Expecting property name enclosed in double quotes"));
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return Err(self.err("Expecting ':' delimiter"));
                    }
                    self.i += 1;
                    let v = self.value()?;
                    map.insert(k, v);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(PyJson::Dict(map));
                        }
                        _ => return Err(self.err("Expecting ',' delimiter")),
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut list = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(PyJson::List(list));
                }
                loop {
                    list.push(self.value()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(PyJson::List(list));
                        }
                        _ => return Err(self.err("Expecting ',' delimiter")),
                    }
                }
            }
            b'"' => Ok(PyJson::Str(self.string()?)),
            _ if self.lit("null") => Ok(PyJson::Null),
            _ if self.lit("true") => Ok(PyJson::Bool(true)),
            _ if self.lit("false") => Ok(PyJson::Bool(false)),
            _ if self.lit("NaN") => Ok(PyJson::Float(f64::NAN)),
            _ if self.lit("Infinity") => Ok(PyJson::Float(f64::INFINITY)),
            _ if self.lit("-Infinity") => Ok(PyJson::Float(f64::NEG_INFINITY)),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.err("Expecting value")),
        }
    }

    fn number(&mut self) -> Result<PyJson> {
        let start = self.i;
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                    self.i += 1;
                }
            }
            _ => return Err(self.err("Expecting value")),
        }
        let mut is_float = false;
        if self.s.get(self.i) == Some(&b'.') && self.s.get(self.i + 1).is_some_and(u8::is_ascii_digit) {
            is_float = true;
            self.i += 1;
            while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                self.i += 1;
            }
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                is_float = true;
                while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                    self.i += 1;
                }
            } else {
                self.i = save;
            }
        }
        let text = &self.src[start..self.i];
        if is_float {
            text.parse().map(PyJson::Float).map_err(|_| self.err("invalid number"))
        } else {
            // `int("-0")` == 0
            let t = if text == "-0" { "0" } else { text };
            Ok(PyJson::Int(t.to_string()))
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let h = self.src.get(self.i..self.i + 4).ok_or_else(|| self.err("Invalid \\uXXXX escape"))?;
        let v = u32::from_str_radix(h, 16).map_err(|_| self.err("Invalid \\uXXXX escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        self.i += 1; // aspas de abertura
        let mut out = String::new();
        loop {
            let rest = &self.src[self.i..];
            let Some(c) = rest.chars().next() else { return Err(self.err("Unterminated string")) };
            match c {
                '"' => {
                    self.i += 1;
                    return Ok(out);
                }
                '\\' => {
                    self.i += 1;
                    let Some(&e) = self.s.get(self.i) else { return Err(self.err("Unterminated string")) };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) && self.src[self.i..].starts_with("\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    self.i = save;
                                    hi
                                }
                            } else {
                                hi
                            };
                            // Surrogate solto vira U+FFFD (o Rust não representa surrogates).
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(self.err("Invalid \\escape")),
                    }
                }
                c if (c as u32) < 0x20 => return Err(self.err("Invalid control character")),
                c => {
                    out.push(c);
                    self.i += c.len_utf8();
                }
            }
        }
    }
}

/// `json.loads`.
pub fn loads(text: &str) -> Result<PyJson> {
    let mut p = Parser { s: text.as_bytes(), src: text, i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != text.len() {
        return Err(p.err("Extra data"));
    }
    Ok(v)
}

/// `repr(float)` do Python.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Dígitos mais curtos que fazem o round-trip, no formato d.ddde±x.
    let sci = format!("{f:e}");
    let (mant, exp) = sci.split_once('e').expect("formato e");
    let exp: i32 = exp.parse().expect("expoente");
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let point = exp + 1; // posição do ponto em relação aos dígitos
        let s = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point >= n {
            format!("{}{}.0", digits, "0".repeat((point - n) as usize))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

fn dump_str(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps(value, ensure_ascii=...)`.
pub fn dumps(v: &PyJson, ensure_ascii: bool) -> String {
    let mut out = String::new();
    dump(v, ensure_ascii, &mut out);
    out
}

fn dump(v: &PyJson, ensure_ascii: bool, out: &mut String) {
    match v {
        PyJson::Null => out.push_str("null"),
        PyJson::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        PyJson::Int(i) => out.push_str(i),
        PyJson::Float(f) => out.push_str(&float_repr(*f)),
        PyJson::Str(s) => dump_str(s, ensure_ascii, out),
        PyJson::List(l) => {
            out.push('[');
            for (i, x) in l.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump(x, ensure_ascii, out);
            }
            out.push(']');
        }
        PyJson::Dict(d) => {
            out.push('{');
            for (i, (k, x)) in d.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                dump_str(k, ensure_ascii, out);
                out.push_str(": ");
                dump(x, ensure_ascii, out);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_float_repr() {
        for (f, s) in [(1.0, "1.0"), (1.5, "1.5"), (1e-5, "1e-05"), (1e16, "1e+16"), (123.456, "123.456"), (0.0001, "0.0001"), (1.5e20, "1.5e+20"), (-2.5, "-2.5")] {
            assert_eq!(float_repr(f), s);
        }
    }

    #[test]
    fn roundtrip_like_python() {
        let v = loads(r#"{"b": 1, "a": [1.50, "é😀"], "b": 2, "n": -0}"#).unwrap();
        assert_eq!(dumps(&v, true), "{\"b\": 2, \"a\": [1.5, \"\\u00e9\\ud83d\\ude00\"], \"n\": 0}");
        assert_eq!(dumps(&v, false), r#"{"b": 2, "a": [1.5, "é😀"], "n": 0}"#);
    }
}
