//! MurmurHash3 (x86, 32 bits) — o `murmurhash3_32` do scikit-learn, usado pelo `hashing`.

/// Hash com sinal (`murmurhash3_bytes_s32`) dos bytes UTF-8.
pub fn murmurhash3_32(data: &[u8], seed: u32) -> i32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h = seed;
    let mut chunks = data.chunks_exact(4);
    for chunk in &mut chunks {
        let mut k = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let tail = chunks.remainder();
    let mut k: u32 = 0;
    if tail.len() >= 3 {
        k ^= (tail[2] as u32) << 16;
    }
    if tail.len() >= 2 {
        k ^= (tail[1] as u32) << 8;
    }
    if !tail.is_empty() {
        k ^= tail[0] as u32;
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }
    h ^= data.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h as i32
}

/// Coluna e sinal de um termo no `FeatureHasher` do scikit-learn.
pub fn hashed_index(term: &str, n_features: usize) -> (usize, f64) {
    let h = murmurhash3_32(term.as_bytes(), 0);
    let index = if h == i32::MIN {
        // abs(-2**31) não cabe em i32: o scikit-learn define o resultado como abaixo.
        (2_147_483_647 - (n_features as i64 - 1)).rem_euclid(n_features as i64) as usize
    } else {
        (h.unsigned_abs() as usize) % n_features
    };
    (index, if h >= 0 { 1.0 } else { -1.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scikit_learn() {
        // sklearn.utils.murmurhash3_32(...) com seed 0 (positive=False).
        assert_eq!(murmurhash3_32(b"foo", 0), -156908512);
        assert_eq!(murmurhash3_32(b"", 0), 0);
        assert_eq!(murmurhash3_32(b"foo", 42), -1322301282);
        assert_eq!(murmurhash3_32("documento".as_bytes(), 0), -1812630357);
        assert_eq!(murmurhash3_32("ação".as_bytes(), 0), -206526165);
        assert_eq!(murmurhash3_32(b"abcd", 0), 1139631978);
        assert_eq!(murmurhash3_32(b"abcde", 0), -392455434);
    }
}
