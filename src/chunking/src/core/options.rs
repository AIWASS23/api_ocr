//! Leitura tipada de `options`. No Python as opções viram kwargs do splitter, e um kwarg
//! desconhecido dá `TypeError`; aqui cada estratégia retira o que entende e
//! [`Options::finish`] recusa o que sobrar.

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::core::splitter::KeepSeparator;
use crate::error::{ChunkError, Result};

pub struct Options {
    strategy: &'static str,
    map: Map<String, Value>,
}

impl Options {
    pub fn new(strategy: &'static str, map: &Map<String, Value>) -> Self {
        let mut map = map.clone();
        // `chunker`/`splitter` são tratados pelo manager (splitter externo).
        map.remove("chunker");
        map.remove("splitter");
        Self { strategy, map }
    }

    fn err(&self, key: &str, expected: &str) -> ChunkError {
        ChunkError::config(format!("{}: option '{key}' must be {expected}", self.strategy))
    }

    pub fn take<T: DeserializeOwned>(&mut self, key: &str) -> Result<Option<T>> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => serde_json::from_value(v).map(Some).map_err(|e| {
                ChunkError::config(format!("{}: option '{key}': {e}", self.strategy))
            }),
        }
    }

    pub fn bool(&mut self, key: &str, default: bool) -> Result<bool> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(default),
            Some(Value::Bool(b)) => Ok(b),
            Some(_) => Err(self.err(key, "a boolean")),
        }
    }

    pub fn string(&mut self, key: &str) -> Result<Option<String>> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(self.err(key, "a string")),
        }
    }

    pub fn usize(&mut self, key: &str) -> Result<Option<usize>> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) if n.as_u64().is_some() => Ok(n.as_u64().map(|v| v as usize)),
            Some(_) => Err(self.err(key, "a non-negative integer")),
        }
    }

    pub fn f64(&mut self, key: &str) -> Result<Option<f64>> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) => Ok(n.as_f64()),
            Some(_) => Err(self.err(key, "a number")),
        }
    }

    pub fn strings(&mut self, key: &str) -> Result<Option<Vec<String>>> {
        self.take::<Vec<String>>(key)
    }

    /// Pares `[["#", "h1"], ...]` (headers_to_split_on).
    pub fn pairs(&mut self, key: &str) -> Result<Option<Vec<(String, String)>>> {
        self.take::<Vec<(String, String)>>(key)
    }

    pub fn keep_separator(&mut self, default: KeepSeparator) -> Result<KeepSeparator> {
        match self.map.remove("keep_separator") {
            None | Some(Value::Null) => Ok(default),
            Some(v) => KeepSeparator::from_value(&v),
        }
    }

    /// Remove opções que o splitter do Python aceita mas que não mudam o texto dos chunks
    /// (ex.: `add_start_index`, que só afeta metadados no LangChain).
    pub fn ignore(&mut self, keys: &[&str]) {
        for k in keys {
            self.map.remove(*k);
        }
    }

    /// Recusa opções não reconhecidas (o `TypeError` de kwarg inesperado do Python).
    pub fn finish(self) -> Result<()> {
        if self.map.is_empty() {
            return Ok(());
        }
        let mut keys: Vec<&String> = self.map.keys().collect();
        keys.sort();
        let names: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
        Err(ChunkError::config(format!("{}: unexpected option(s): {}", self.strategy, names.join(", "))))
    }
}
