//! Controle de admissão por memória: não aceita novo OCR quando o processo (ou o
//! contêiner) já está perto do limite. O motor de OCR não tem teto de memória por
//! documento; isto evita que uma rajada de jobs pesados leve o processo ao OOM-kill.

use std::fs;

/// `(usado, limite)` em bytes.
pub type Usage = (u64, u64);

pub struct MemoryGuard {
    watermark_pct: u8,
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Memória "de trabalho" do cgroup v2 (uso menos cache de arquivo recuperável),
/// se houver limite; senão, a do host (`MemTotal - MemAvailable`).
pub fn system_usage() -> Option<Usage> {
    if let Some(limit) = read_u64("/sys/fs/cgroup/memory.max") {
        // "max" não é número => sem limite de cgroup.
        let current = read_u64("/sys/fs/cgroup/memory.current")?;
        let inactive = fs::read_to_string("/sys/fs/cgroup/memory.stat")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("inactive_file ").and_then(|v| v.trim().parse::<u64>().ok()))
            })
            .unwrap_or(0);
        return Some((current.saturating_sub(inactive), limit));
    }
    let info = fs::read_to_string("/proc/meminfo").ok()?;
    let kb = |key: &str| -> Option<u64> {
        info.lines()
            .find_map(|l| l.strip_prefix(key))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .ok()
    };
    let (total, avail) = (kb("MemTotal:")?, kb("MemAvailable:")?);
    Some((total.saturating_sub(avail) * 1024, total * 1024))
}

/// `used * 100 >= limit * pct`, sem estourar.
pub fn over(usage: Usage, pct: u8) -> bool {
    let (used, limit) = usage;
    limit > 0 && (used as u128) * 100 >= (limit as u128) * (pct as u128)
}

impl MemoryGuard {
    pub fn new(watermark_pct: u8) -> Self {
        Self { watermark_pct: watermark_pct.clamp(1, 100) }
    }

    /// `true` quando não deve aceitar novo trabalho pesado. Se não der para medir, libera.
    pub fn is_under_pressure(&self) -> bool {
        self.watermark_pct < 100 && system_usage().is_some_and(|u| over(u, self.watermark_pct))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watermark_math() {
        assert!(over((85, 100), 85));
        assert!(!over((84, 100), 85));
        assert!(over((u64::MAX, u64::MAX), 100));
        assert!(!over((1, 0), 85), "sem limite conhecido não bloqueia");
    }

    #[test]
    fn host_usage_is_readable_and_sane() {
        let (used, limit) = system_usage().expect("lê /proc/meminfo ou cgroup");
        assert!(limit > 0 && used <= limit);
    }

    #[test]
    fn guard_100_disables_and_1_trips() {
        assert!(!MemoryGuard::new(100).is_under_pressure());
        assert!(MemoryGuard::new(1).is_under_pressure());
    }
}
