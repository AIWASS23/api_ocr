//! Leitor mínimo de `pytorch_model.bin` (formato zip do `torch.save`, pickle protocolo 2-4) para
//! `state_dict`s de tensores. Serve às camadas `Dense` do sentence-transformers que só existem
//! nesse formato (ex.: `hkunlp/instructor-*`). Não executa código: só os opcodes de dados e as
//! reconstruções de tensor conhecidas (`_rebuild_tensor_v2`, `_rebuild_parameter`,
//! `OrderedDict`); qualquer outra chamada é recusada.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::error::{EmbeddingError, Result};

#[derive(Clone, Debug)]
enum Val {
    None,
    Bool(#[allow(dead_code)] bool),
    Int(i64),
    Float(#[allow(dead_code)] f64),
    Str(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global(String, String),
    Persistent(Box<Val>),
    Tensor(TensorRef),
    Mark,
}

#[derive(Clone, Debug)]
struct TensorRef {
    storage: String,
    dtype: String,
    offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

/// Tensor lido: forma e valores em `f32`, contíguos por linha.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

fn err(msg: impl std::fmt::Display) -> EmbeddingError {
    EmbeddingError::model(format!("pytorch_model.bin: {msg}"))
}

fn as_usize_vec(v: &Val) -> Result<Vec<usize>> {
    match v {
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .map(|x| match x {
                Val::Int(i) if *i >= 0 => Ok(*i as usize),
                _ => Err(err("tensor shape/stride must be non-negative integers")),
            })
            .collect(),
        _ => Err(err("tensor shape/stride must be a tuple")),
    }
}

fn reduce(callable: Val, args: Val) -> Result<Val> {
    let Val::Global(module, name) = &callable else { return Err(err("REDUCE on a non-global")) };
    let args = match args {
        Val::Tuple(a) => a,
        _ => return Err(err("REDUCE arguments must be a tuple")),
    };
    match (module.as_str(), name.as_str()) {
        ("collections", "OrderedDict") => Ok(Val::Dict(Vec::new())),
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            let storage = match args.first() {
                Some(Val::Persistent(p)) => match p.as_ref() {
                    // ('storage', <tipo>, chave, local, numel)
                    Val::Tuple(t) if t.len() >= 3 => {
                        let dtype = match &t[1] {
                            Val::Global(_, n) => n.clone(),
                            Val::Str(s) => s.clone(),
                            _ => return Err(err("unknown storage type")),
                        };
                        let key = match &t[2] {
                            Val::Str(s) => s.clone(),
                            Val::Int(i) => i.to_string(),
                            _ => return Err(err("unknown storage key")),
                        };
                        (key, dtype)
                    }
                    _ => return Err(err("unexpected persistent id")),
                },
                _ => return Err(err("tensor without storage")),
            };
            let offset = match args.get(1) {
                Some(Val::Int(i)) if *i >= 0 => *i as usize,
                _ => return Err(err("invalid storage offset")),
            };
            let size = as_usize_vec(args.get(2).ok_or_else(|| err("tensor without size"))?)?;
            let stride = as_usize_vec(args.get(3).ok_or_else(|| err("tensor without stride"))?)?;
            Ok(Val::Tensor(TensorRef { storage: storage.0, dtype: storage.1, offset, size, stride }))
        }
        // `torch.nn.Parameter`: o tensor de dentro.
        ("torch._utils", "_rebuild_parameter") | ("torch._utils", "_rebuild_parameter_with_state") => {
            args.into_iter().next().ok_or_else(|| err("parameter without data"))
        }
        (m, n) => Err(err(format!("refusing to call {m}.{n} (only tensors are supported)"))),
    }
}

fn take<'a>(data: &'a [u8], i: &mut usize, n: usize) -> Result<&'a [u8]> {
    let s = data.get(*i..*i + n).ok_or_else(|| err("truncated pickle"))?;
    *i += n;
    Ok(s)
}

fn pop(stack: &mut Vec<Val>) -> Result<Val> {
    stack.pop().ok_or_else(|| err("pickle stack underflow"))
}

fn pop_mark(stack: &mut Vec<Val>) -> Result<Vec<Val>> {
    let pos = stack.iter().rposition(|v| matches!(v, Val::Mark)).ok_or_else(|| err("MARK not found"))?;
    let items = stack.split_off(pos + 1);
    stack.pop();
    Ok(items)
}

fn read_line(data: &[u8], i: &mut usize) -> Result<String> {
    let start = *i;
    let end = data[start..].iter().position(|&c| c == b'\n').ok_or_else(|| err("GLOBAL without newline"))? + start;
    *i = end + 1;
    Ok(String::from_utf8_lossy(&data[start..end]).into_owned())
}

fn arr<const N: usize>(b: &[u8]) -> Result<[u8; N]> {
    b.try_into().map_err(|_| err("truncated pickle"))
}

/// Interpreta o pickle e devolve o objeto do topo.
fn unpickle(data: &[u8]) -> Result<Val> {
    let mut stack: Vec<Val> = Vec::new();
    let mut memo: HashMap<u32, Val> = HashMap::new();
    let mut i = 0usize;
    loop {
        let op = *data.get(i).ok_or_else(|| err("pickle without STOP"))?;
        i += 1;
        match op {
            // PROTO, FRAME
            0x80 => {
                take(data, &mut i, 1)?;
            }
            0x95 => {
                take(data, &mut i, 8)?;
            }
            b'.' => return pop(&mut stack), // STOP
            b'}' => stack.push(Val::Dict(Vec::new())),
            b']' => stack.push(Val::List(Vec::new())),
            b')' => stack.push(Val::Tuple(Vec::new())),
            b'(' => stack.push(Val::Mark),
            b'N' => stack.push(Val::None),
            0x88 => stack.push(Val::Bool(true)),
            0x89 => stack.push(Val::Bool(false)),
            b'K' => {
                let b = take(data, &mut i, 1)?[0];
                stack.push(Val::Int(b as i64));
            }
            b'M' => {
                let b = take(data, &mut i, 2)?;
                stack.push(Val::Int(u16::from_le_bytes([b[0], b[1]]) as i64));
            }
            b'J' => {
                let b = take(data, &mut i, 4)?;
                stack.push(Val::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64));
            }
            0x8a => {
                // LONG1: inteiro little-endian com sinal, n bytes.
                let n = take(data, &mut i, 1)?[0] as usize;
                let b = take(data, &mut i, n)?;
                let mut v: i64 = 0;
                for (k, byte) in b.iter().enumerate().take(8) {
                    v |= (*byte as i64) << (8 * k);
                }
                if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                    v -= 1i64 << (8 * n);
                }
                stack.push(Val::Int(v));
            }
            b'G' => {
                let b = take(data, &mut i, 8)?;
                stack.push(Val::Float(f64::from_be_bytes(arr::<8>(b)?)));
            }
            b'X' | 0x8c | 0x8d => {
                let n = match op {
                    b'X' => u32::from_le_bytes(arr::<4>(take(data, &mut i, 4)?)?) as usize,
                    0x8c => take(data, &mut i, 1)?[0] as usize,
                    _ => u64::from_le_bytes(arr::<8>(take(data, &mut i, 8)?)?) as usize,
                };
                let s = std::str::from_utf8(take(data, &mut i, n)?).map_err(|_| err("invalid UTF-8 string"))?;
                stack.push(Val::Str(s.to_string()));
            }
            b'c' => {
                let module = read_line(data, &mut i)?;
                let name = read_line(data, &mut i)?;
                stack.push(Val::Global(module, name));
            }
            0x93 => {
                // STACK_GLOBAL
                let name = pop(&mut stack)?;
                let module = pop(&mut stack)?;
                match (module, name) {
                    (Val::Str(m), Val::Str(n)) => stack.push(Val::Global(m, n)),
                    _ => return Err(err("STACK_GLOBAL needs strings")),
                }
            }
            b'q' => {
                let k = take(data, &mut i, 1)?[0] as u32;
                memo.insert(k, stack.last().cloned().ok_or_else(|| err("BINPUT on empty stack"))?);
            }
            b'r' => {
                let k = u32::from_le_bytes(arr::<4>(take(data, &mut i, 4)?)?);
                memo.insert(k, stack.last().cloned().ok_or_else(|| err("LONG_BINPUT on empty stack"))?);
            }
            0x94 => {
                let k = memo.len() as u32;
                memo.insert(k, stack.last().cloned().ok_or_else(|| err("MEMOIZE on empty stack"))?);
            }
            b'h' => {
                let k = take(data, &mut i, 1)?[0] as u32;
                stack.push(memo.get(&k).cloned().ok_or_else(|| err("BINGET of unknown memo"))?);
            }
            b'j' => {
                let k = u32::from_le_bytes(arr::<4>(take(data, &mut i, 4)?)?);
                stack.push(memo.get(&k).cloned().ok_or_else(|| err("LONG_BINGET of unknown memo"))?);
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(Val::Tuple(items));
            }
            0x85..=0x87 => {
                let n = (op - 0x84) as usize;
                if stack.len() < n {
                    return Err(err("tuple underflow"));
                }
                let items = stack.split_off(stack.len() - n);
                stack.push(Val::Tuple(items));
            }
            b'Q' => {
                let pid = pop(&mut stack)?;
                stack.push(Val::Persistent(Box::new(pid)));
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(reduce(callable, args)?);
            }
            0x81 => {
                // NEWOBJ: cls.__new__(cls, *args) — tratado como REDUCE.
                let args = pop(&mut stack)?;
                let cls = pop(&mut stack)?;
                stack.push(reduce(cls, args)?);
            }
            b'b' => {
                // BUILD: estado do objeto (ex.: `_metadata` do state_dict); ignorado.
                pop(&mut stack)?;
            }
            b's' => {
                let value = pop(&mut stack)?;
                let key = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::Dict(d)) => d.push((key, value)),
                    _ => return Err(err("SETITEM on a non-dict")),
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::Dict(d)) => {
                        let mut it = items.into_iter();
                        while let (Some(k), Some(v)) = (it.next(), it.next()) {
                            d.push((k, v));
                        }
                    }
                    _ => return Err(err("SETITEMS on a non-dict")),
                }
            }
            b'a' => {
                let v = pop(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::List(l)) => l.push(v),
                    _ => return Err(err("APPEND on a non-list")),
                }
            }
            b'e' => {
                let items = pop_mark(&mut stack)?;
                match stack.last_mut() {
                    Some(Val::List(l)) => l.extend(items),
                    _ => return Err(err("APPENDS on a non-list")),
                }
            }
            other => return Err(err(format!("unsupported pickle opcode 0x{other:02x}"))),
        }
    }
}

fn element_size(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "FloatStorage" | "float32" => 4,
        "DoubleStorage" | "float64" => 8,
        "HalfStorage" | "BFloat16Storage" | "float16" | "bfloat16" => 2,
        other => return Err(err(format!("unsupported tensor storage {other}"))),
    })
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let v = match exp {
        0 if frac == 0 => sign,
        0 => {
            // Subnormal: normaliza.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(v)
}

fn read_element(bytes: &[u8], dtype: &str, index: usize) -> Result<f32> {
    let size = element_size(dtype)?;
    let b = bytes.get(index * size..(index + 1) * size).ok_or_else(|| err("tensor data out of range"))?;
    Ok(match dtype {
        "FloatStorage" | "float32" => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        "DoubleStorage" | "float64" => f64::from_le_bytes(b.try_into().map_err(|_| err("f64"))?) as f32,
        "HalfStorage" | "float16" => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
        _ => f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
    })
}

/// Lê todos os tensores de um `pytorch_model.bin` (formato zip).
pub fn load_state_dict(path: &Path) -> Result<HashMap<String, TensorData>> {
    let file = std::fs::File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let mut zip = zip::ZipArchive::new(file).map_err(|_| err("not a zip-format checkpoint (legacy torch.save formats are not supported)"))?;
    let pkl_name = zip
        .file_names()
        .filter_map(|n| n.ok().map(|c| c.into_owned()))
        .find(|n| n.ends_with("/data.pkl") || n == "data.pkl")
        .ok_or_else(|| err("data.pkl not found"))?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    let mut pkl = Vec::new();
    zip.by_name(&pkl_name).map_err(err)?.read_to_end(&mut pkl).map_err(err)?;
    let top = unpickle(&pkl)?;
    let Val::Dict(entries) = top else { return Err(err("top-level object is not a state_dict")) };
    let mut storages: HashMap<String, Vec<u8>> = HashMap::new();
    let mut out = HashMap::new();
    for (k, v) in entries {
        let (Val::Str(name), Val::Tensor(t)) = (k, v) else { continue };
        if !storages.contains_key(&t.storage) {
            let mut bytes = Vec::new();
            zip.by_name(&format!("{prefix}data/{}", t.storage)).map_err(err)?.read_to_end(&mut bytes).map_err(err)?;
            storages.insert(t.storage.clone(), bytes);
        }
        let bytes = &storages[&t.storage];
        let numel: usize = t.size.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut idx = vec![0usize; t.size.len()];
        for _ in 0..numel {
            let pos = t.offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
            data.push(read_element(bytes, &t.dtype, pos)?);
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < t.size[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out.insert(name, TensorData { shape: t.size, data });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert!((half_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }

    #[test]
    fn refuses_arbitrary_calls() {
        // cos\nsystem\n + (S'ls' tuple) + REDUCE
        let pickle = b"\x80\x02cos\nsystem\nX\x02\x00\x00\x00ls\x85R.";
        let e = unpickle(pickle).unwrap_err().to_string();
        assert!(e.contains("refusing to call os.system"), "{e}");
    }

    #[test]
    fn ordered_dict_of_tensors() {
        // OrderedDict() ; {'w': _rebuild_tensor_v2(persistent(('storage', torch.FloatStorage, '0', 'cpu', 2)), 0, (2,), (1,), False, OrderedDict())}
        let p: &[u8] = b"\x80\x02ccollections\nOrderedDict\nq\x00)Rq\x01(X\x01\x00\x00\x00wq\x02ctorch._utils\n_rebuild_tensor_v2\nq\x03((X\x07\x00\x00\x00storageq\x04ctorch\nFloatStorage\nq\x05X\x01\x00\x00\x000q\x06X\x03\x00\x00\x00cpuq\x07K\x02tq\x08QK\x00K\x02\x85q\tK\x01\x85q\n\x89h\x00)Rq\x0btq\x0cRq\ru.";
        let top = unpickle(p).unwrap();
        let Val::Dict(d) = top else { panic!("dict") };
        assert!(matches!(&d[0].1, Val::Tensor(t) if t.size == vec![2] && t.storage == "0" && t.dtype == "FloatStorage"));
    }
}
