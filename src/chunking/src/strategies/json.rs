//! `json` — port do `RecursiveJsonSplitter`: percorre o objeto e agrupa chaves em pedaços de
//! até `chunk_size` caracteres de `json.dumps`, preservando o caminho de cada valor. Lista no
//! topo vira `{"items": [...]}`; texto que não é JSON cai no corte recursivo. Depois o tamanho
//! é imposto (`_enforce_size`).
//!
//! Opções: `convert_lists` (`false`: listas viram objetos `{"0": ...}`), `ensure_ascii` (`true`),
//! `min_chunk_size` (`max(chunk_size - 200, 50)`).

use indexmap::IndexMap;

use crate::chunker::{Chunker, Sizing};
use crate::core::options::Options;
use crate::core::pyjson::{PyJson, dumps, loads};
use crate::core::pystr::{self, strip};
use crate::error::Result;
use crate::strategies::recursive::enforce_size;

pub struct ChunkJson {
    sizing: Sizing,
    min_chunk_size: usize,
    convert_lists: bool,
    ensure_ascii: bool,
}

type Dict = IndexMap<String, PyJson>;

fn size(d: &PyJson) -> usize {
    pystr::len(&dumps(d, true))
}

/// `_set_nested_dict`: cria os objetos do caminho com `setdefault`.
fn set_nested(d: &mut Dict, path: &[String], value: PyJson) {
    let (last, init) = path.split_last().expect("caminho não vazio");
    let mut cur = d;
    for key in init {
        let entry = cur.entry(key.clone()).or_insert_with(|| PyJson::Dict(IndexMap::new()));
        if !matches!(entry, PyJson::Dict(_)) {
            *entry = PyJson::Dict(IndexMap::new());
        }
        let PyJson::Dict(next) = entry else { unreachable!() };
        cur = next;
    }
    cur.insert(last.clone(), value);
}

fn list_to_dict(v: PyJson) -> PyJson {
    match v {
        PyJson::Dict(d) => PyJson::Dict(d.into_iter().map(|(k, v)| (k, list_to_dict(v))).collect()),
        PyJson::List(l) => PyJson::Dict(l.into_iter().enumerate().map(|(i, v)| (i.to_string(), list_to_dict(v))).collect()),
        other => other,
    }
}

impl ChunkJson {
    pub fn new(sizing: Sizing, mut o: Options) -> Result<Self> {
        let convert_lists = o.bool("convert_lists", false)?;
        let ensure_ascii = o.bool("ensure_ascii", true)?;
        let min_chunk_size = o.usize("min_chunk_size")?.unwrap_or_else(|| sizing.chunk_size.saturating_sub(200).max(50));
        o.finish()?;
        Ok(Self { sizing, min_chunk_size, convert_lists, ensure_ascii })
    }

    /// `_json_split`.
    fn json_split(&self, data: &PyJson, path: &[String], chunks: &mut Vec<Dict>) {
        match data {
            PyJson::Dict(d) if !d.is_empty() => {
                for (key, value) in d {
                    let mut new_path = path.to_vec();
                    new_path.push(key.clone());
                    let chunk_size = size(&PyJson::Dict(chunks.last().cloned().unwrap_or_default()));
                    let mut single = IndexMap::new();
                    single.insert(key.clone(), value.clone());
                    let item = size(&PyJson::Dict(single));
                    let remaining = self.sizing.chunk_size as i64 - chunk_size as i64;
                    if (item as i64) < remaining {
                        set_nested(chunks.last_mut().expect("sempre há um chunk"), &new_path, value.clone());
                    } else {
                        if chunk_size >= self.min_chunk_size {
                            chunks.push(IndexMap::new());
                        }
                        self.json_split(value, &new_path, chunks);
                    }
                }
            }
            _ if !path.is_empty() => set_nested(chunks.last_mut().expect("sempre há um chunk"), path, data.clone()),
            _ => {}
        }
    }

    /// `RecursiveJsonSplitter.split_text` sobre um valor já lido.
    pub fn split_value(&self, value: PyJson) -> Vec<String> {
        let value = if self.convert_lists { list_to_dict(value) } else { value };
        let mut chunks: Vec<Dict> = vec![IndexMap::new()];
        self.json_split(&value, &[], &mut chunks);
        if chunks.last().is_some_and(IndexMap::is_empty) {
            chunks.pop();
        }
        chunks.into_iter().map(|c| dumps(&PyJson::Dict(c), self.ensure_ascii)).collect()
    }
}

impl Chunker for ChunkJson {
    crate::sizing_accessors!("json");

    fn chunk(&self, text: &str) -> Result<Vec<String>> {
        if strip(text).is_empty() {
            return Ok(Vec::new());
        }
        let Ok(value) = loads(text) else {
            return enforce_size(&[text.to_string()], self.sizing);
        };
        let value = match value {
            PyJson::Dict(d) => PyJson::Dict(d),
            other => {
                let mut wrap = IndexMap::new();
                wrap.insert("items".to_string(), other);
                PyJson::Dict(wrap)
            }
        };
        enforce_size(&self.split_value(value), self.sizing)
    }
}
