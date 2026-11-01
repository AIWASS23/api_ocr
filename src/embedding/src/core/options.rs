//! Leitura tipada de `options`. No Python as opções viram kwargs do modelo/vetorizador e um
//! kwarg desconhecido dá `TypeError`; aqui cada estratégia retira o que entende e
//! [`Options::finish`] recusa o que sobrar.

use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::error::{EmbeddingError, Result};

pub struct Options {
    strategy: &'static str,
    map: Map<String, Value>,
}

impl Options {
    pub fn new(strategy: &'static str, map: &Map<String, Value>) -> Self {
        Self { strategy, map: map.clone() }
    }

    fn err(&self, key: &str, expected: &str) -> EmbeddingError {
        EmbeddingError::config(format!("{}: option '{key}' must be {expected}", self.strategy))
    }

    pub fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn take<T: DeserializeOwned>(&mut self, key: &str) -> Result<Option<T>> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => serde_json::from_value(v).map(Some).map_err(|e| self.err(key, &format!("valid ({e})"))),
        }
    }

    pub fn raw(&mut self, key: &str) -> Option<Value> {
        self.map.remove(key).filter(|v| !v.is_null())
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
            Some(v) => v.as_u64().map(|n| Some(n as usize)).ok_or_else(|| self.err(key, "a non-negative integer")),
        }
    }

    pub fn f64(&mut self, key: &str) -> Result<Option<f64>> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_f64().map(Some).ok_or_else(|| self.err(key, "a number")),
        }
    }

    /// Submapa (`model_options`, `encode_options`, `embed_options`...).
    pub fn object(&mut self, key: &str) -> Result<Map<String, Value>> {
        match self.map.remove(key) {
            None | Some(Value::Null) => Ok(Map::new()),
            Some(Value::Object(m)) => Ok(m),
            Some(_) => Err(self.err(key, "a mapping")),
        }
    }

    /// Opções aceitas por compatibilidade com o Python e sem efeito aqui.
    pub fn ignore(&mut self, keys: &[&str]) {
        for k in keys {
            self.map.remove(*k);
        }
    }

    /// Erro se sobrou alguma opção desconhecida.
    pub fn finish(self) -> Result<()> {
        if self.map.is_empty() {
            return Ok(());
        }
        let mut keys: Vec<&str> = self.map.keys().map(String::as_str).collect();
        keys.sort_unstable();
        Err(EmbeddingError::config(format!("{}: unknown option(s): {}", self.strategy, keys.join(", "))))
    }
}
