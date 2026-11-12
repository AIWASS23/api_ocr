//! Datas dos metadados (`_timestamp` do Python): número (segundos Unix) ou texto ISO 8601
//! (`2024-05-01`, `2024-05-01T10:00:00`, com fração, `Z` ou `±HH:MM`).
//!
//! Diferença intencional: data/hora sem fuso vale como UTC (no Python, como hora local do
//! servidor), para o resultado não depender da máquina.

use serde_json::Value;

/// Dias desde 1970-01-01 (algoritmo de Howard Hinnant).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn num(s: &str) -> Option<i64> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok()).flatten()
}

/// Segundos Unix de um texto ISO 8601.
pub fn parse_iso(s: &str) -> Option<f64> {
    let s = s.trim();
    let date = s.get(..10)?;
    let parts: Vec<&str> = date.split('-').collect();
    let [y, m, d] = parts.as_slice() else { return None };
    let (y, m, d) = (num(y)?, num(m)?, num(d)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let mut secs = days_from_civil(y, m, d) as f64 * 86_400.0;
    let mut rest = &s[10..];
    if rest.is_empty() {
        return Some(secs);
    }
    if !rest.starts_with(['T', ' ']) {
        return None;
    }
    rest = &rest[1..];
    // Fuso no fim: Z ou ±HH[:MM].
    let mut offset = 0.0;
    if let Some(r) = rest.strip_suffix('Z') {
        rest = r;
    } else if let Some(pos) = rest.rfind(['+', '-']) {
        let (clock, tz) = rest.split_at(pos);
        let sign = if tz.starts_with('-') { -1.0 } else { 1.0 };
        let hm: Vec<&str> = tz[1..].split(':').collect();
        let h = num(hm.first()?)? as f64;
        let mi = hm.get(1).and_then(|x| num(x)).unwrap_or(0) as f64;
        offset = sign * (h * 3600.0 + mi * 60.0);
        rest = clock;
    }
    let hms: Vec<&str> = rest.split(':').collect();
    let h = num(hms.first()?)?;
    let mi = hms.get(1).map_or(Some(0), |x| num(x))?;
    let sec: f64 = match hms.get(2) {
        None => 0.0,
        Some(x) => x.parse().ok().filter(|v: &f64| v.is_finite() && *v >= 0.0)?,
    };
    secs += (h * 3600 + mi * 60) as f64 + sec;
    Some(secs - offset)
}

pub fn timestamp(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => parse_iso(s),
        _ => None,
    }
}

pub fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_dates() {
        assert_eq!(parse_iso("1970-01-02"), Some(86_400.0));
        assert_eq!(parse_iso("2024-01-01T00:00:00Z"), Some(1_704_067_200.0));
        assert_eq!(parse_iso("2024-01-01T03:00:00+03:00"), Some(1_704_067_200.0));
        assert_eq!(parse_iso("2024-01-01 00:00:01.5"), Some(1_704_067_201.5));
        assert_eq!(parse_iso("ontem"), None);
    }
}
