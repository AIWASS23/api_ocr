//! Agendamento de arquivos: arquivos leves rodam em paralelo; arquivos pesados
//! rodam um por vez, porque o `docling` já paraleliza por página dentro deles
//! (pool de workers para PDFs de 6+ páginas) e dividir os núcleos entre vários
//! pesados só gasta RAM/VRAM.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

/// Limiar padrão de páginas: o mesmo de `DOCLING_RS_PDF_PARALLEL_MIN` no docling,
/// a partir do qual um PDF passa a usar o pool interno de workers.
pub const DEFAULT_HEAVY_PAGES: usize = 6;
/// Arquivos maiores que isso (bytes) são tratados como pesados, qualquer formato.
pub const DEFAULT_HEAVY_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Arquivos leves processados ao mesmo tempo. `0` = automático
    /// (núcleos lógicos / 4, entre 1 e 4). `1` desliga o paralelismo.
    pub max_light_files: usize,
    /// PDFs com esse número de páginas (ou mais) são pesados.
    pub heavy_pages: usize,
    /// Arquivos com esse tamanho (ou mais) são pesados.
    pub heavy_bytes: u64,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_light_files: 0,
            heavy_pages: DEFAULT_HEAVY_PAGES,
            heavy_bytes: DEFAULT_HEAVY_BYTES,
        }
    }
}

impl ParallelConfig {
    pub fn light_workers(&self) -> usize {
        if self.max_light_files > 0 {
            return self.max_light_files;
        }
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        (cores / 4).clamp(1, 4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weight {
    Light,
    Heavy,
}

/// Classifica um arquivo sem convertê-lo.
pub fn classify(path: &Path, cfg: &ParallelConfig) -> Weight {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    // Transcrição de áudio é sempre cara.
    if ext == "mp3" {
        return Weight::Heavy;
    }

    let size = std::fs::metadata(path).map_or(0, |m| m.len());
    if size >= cfg.heavy_bytes {
        return Weight::Heavy;
    }

    if ext == "pdf"
        && let Ok(bytes) = std::fs::read(path)
            && let Ok(pages) = docling::pdf_page_count(&bytes, None)
            && pages >= cfg.heavy_pages
        {
            return Weight::Heavy;
        }

    Weight::Light
}

/// Executa `job` para cada item usando até `workers` threads e devolve os
/// resultados **na ordem dos itens**.
pub fn run_parallel<T, R>(items: &[T], workers: usize, job: impl Fn(&T) -> R + Sync) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    let workers = workers.clamp(1, items.len().max(1));
    if workers == 1 {
        return items.iter().map(&job).collect();
    }

    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    let result = job(item);
                    *slots[i].lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
                }
            });
        }
    });

    slots
        .into_iter()
        .map(|s| {
            s.into_inner()
                .unwrap_or_else(PoisonError::into_inner)
                .expect("todo item processado")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[test]
    fn keeps_input_order_and_runs_concurrently() {
        let items: Vec<u64> = (0..8).collect();
        let running = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);

        let out = run_parallel(&items, 4, |&i| {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            // itens iniciais mais lentos: a ordem de término difere da de entrada
            std::thread::sleep(Duration::from_millis(40 - i * 4));
            running.fetch_sub(1, Ordering::SeqCst);
            i * 10
        });

        assert_eq!(out, [0, 10, 20, 30, 40, 50, 60, 70]);
        assert!(peak.load(Ordering::SeqCst) > 1, "não houve concorrência");
        assert!(peak.load(Ordering::SeqCst) <= 4);
    }

    #[test]
    fn single_worker_and_empty_input() {
        assert_eq!(run_parallel(&[1, 2, 3], 1, |x| x + 1), [2, 3, 4]);
        assert!(run_parallel::<u8, u8>(&[], 4, |x| *x).is_empty());
    }

    #[test]
    fn classifies_by_size_and_extension() {
        let dir = std::env::temp_dir().join(format!("dp_sched_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let small = dir.join("a.docx");
        let big = dir.join("b.xlsx");
        let audio = dir.join("c.mp3");
        std::fs::write(&small, b"x").unwrap();
        std::fs::write(&big, vec![0u8; 2048]).unwrap();
        std::fs::write(&audio, b"x").unwrap();

        let cfg = ParallelConfig {
            heavy_bytes: 1024,
            ..Default::default()
        };
        assert_eq!(classify(&small, &cfg), Weight::Light);
        assert_eq!(classify(&big, &cfg), Weight::Heavy);
        assert_eq!(classify(&audio, &cfg), Weight::Heavy);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
