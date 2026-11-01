//! Worker de OCR: cada documento é inspecionado e convertido num PROCESSO separado. O chunking
//! (`/v1/chunk`) usa o mesmo worker, com o subcomando `chunk-worker` e o mesmo confinamento.
//!
//! Por quê: os parsers (PDF, Office, imagens, ONNX Runtime) são o código mais exposto do
//! servidor. Rodando dentro da API, um bug explorável num deles daria acesso à memória onde
//! estão `JWT_SECRET`, `PASSWORD_PEPPER` e a conexão com o banco. No worker:
//!
//! * o ambiente é zerado e só recebe uma lista fixa de variáveis (nenhum segredo, nenhuma URL);
//! * Landlock próprio: lê só sistema/modelos, grava só no diretório do PRÓPRIO job (não vê os
//!   uploads de outros usuários), nenhuma porta TCP;
//! * cache de grafos ONNX: o confiável (gravado só pelo warm-up, a partir de um documento
//!   embutido no binário) é somente leitura; cada job enxerga uma cópia privada (links) dentro do
//!   próprio diretório, onde o motor pode gravar à vontade — e tudo some com o job. Um worker
//!   comprometido não envenena o cache dos próximos. (Não dá para simplesmente negar a escrita: o
//!   ONNX Runtime derruba o processo com SIGSEGV quando não consegue gravar o grafo otimizado.);
//! * Landlock "scopes": não sinaliza nem conecta (socket abstrato) fora do próprio domínio;
//! * seccomp: sem `execve`, sem sockets de rede, sem ptrace/montagem/namespaces/eBPF...;
//! * `rlimit` de CPU, tamanho de arquivo e descritores;
//! * memória: um VIGIA na API mede a memória não recuperável do contêiner e o RSS de cada worker
//!   a cada 10 ms e mata (SIGKILL) o maior worker antes que o cgroup chegue ao limite. Não dá
//!   para deixar isso com o kernel: no Kubernetes (cgroup v2, `memory.oom.group=1`) o OOM mata o
//!   contêiner INTEIRO, API junto. O `oom_score_adj=1000` cobre o caso de OOM fora de contêiner.
//!   (Teto de espaço de endereçamento é opcional: o ONNX Runtime reserva ~7 GB de endereços com
//!   qualquer documento e TRAVA, em vez de falhar, quando uma reserva é negada.);
//! * não-depurável e sem core dump (um core teria o conteúdo do documento);
//! * um processo novo por documento: um job comprometido não contamina o próximo;
//! * prazo estourado ou cliente desconectado => SIGKILL (antes, a thread seguia rodando).
//!
//! Protocolo: a API executa `<binário> ocr-worker --file <caminho> ...` (ou `chunk-worker`); o
//! worker escreve UMA resposta JSON (`ocr::Reply` / `chunk::Reply`) no stdout. O stdout e o
//! stderr são lidos com teto de tamanho.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once, PoisonError};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

use crate::chunk;
use crate::config::Config;
use crate::hardening::{self, Limits, Profile};
use crate::ocr::{self, Reply};
use crate::sandbox::{self, Mode, Outcome};

/// Maior resposta aceita do worker (texto extraído).
const MAX_REPLY_BYTES: usize = 64 * 1024 * 1024;
/// Quanto do stderr do worker vai para o log da API.
const MAX_STDERR_BYTES: usize = 8 * 1024;
const WARMUP_TIMEOUT: Duration = Duration::from_secs(900);
/// Documento sintético (texto, parágrafos e tabela "escaneados") usado pelo warm-up: passa pelos
/// mesmos modelos e variantes de sessão que um upload real (layout, OCR, TableFormer).
const WARMUP_DOC: &[u8] = include_bytes!("../assets/warmup.pdf");
/// Cache privado de grafos do job (o nome enviado pelo cliente nunca começa com ponto).
const PRIVATE_GRAPHS: &str = ".graph-cache";

#[derive(Debug, thiserror::Error)]
pub enum WorkerFailure {
    #[error("prazo esgotado")]
    Timeout,
    #[error("falha ao iniciar o worker: {0}")]
    Spawn(std::io::Error),
    #[error("worker terminou por sinal {0}")]
    Signaled(i32),
    #[error("worker saiu com código {0}")]
    Exited(i32),
    #[error("resposta do worker maior que o permitido")]
    ReplyTooLarge,
    #[error("resposta do worker inválida")]
    BadReply,
    #[error("worker morto por excesso de memória ({0})")]
    MemoryLimit(&'static str),
}

/// Tipo de job, para o nome dos eventos de segurança.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobKind {
    Ocr,
    Chunk,
    Embedding,
}

impl WorkerFailure {
    /// Nome do evento de segurança. Morte por sinal (SIGSYS do seccomp, SIGSEGV, SIGKILL do
    /// limite de CPU...) é o que se espera ver quando uma entrada tenta explorar o parser.
    pub fn event_kind(&self, job: JobKind) -> &'static str {
        match (job, self) {
            (JobKind::Ocr, WorkerFailure::Signaled(_)) => "ocr_worker_crashed",
            (JobKind::Ocr, WorkerFailure::ReplyTooLarge | WorkerFailure::BadReply) => "ocr_worker_bad_reply",
            (JobKind::Ocr, WorkerFailure::MemoryLimit(_)) => "ocr_worker_memory_killed",
            (JobKind::Ocr, _) => "ocr_worker_failed",
            (JobKind::Chunk, WorkerFailure::Signaled(_)) => "chunk_worker_crashed",
            (JobKind::Chunk, WorkerFailure::ReplyTooLarge | WorkerFailure::BadReply) => "chunk_worker_bad_reply",
            (JobKind::Chunk, WorkerFailure::MemoryLimit(_)) => "chunk_worker_memory_killed",
            (JobKind::Chunk, _) => "chunk_worker_failed",
            (JobKind::Embedding, WorkerFailure::Signaled(_)) => "embedding_worker_crashed",
            (JobKind::Embedding, WorkerFailure::ReplyTooLarge | WorkerFailure::BadReply) => "embedding_worker_bad_reply",
            (JobKind::Embedding, WorkerFailure::MemoryLimit(_)) => "embedding_worker_memory_killed",
            (JobKind::Embedding, _) => "embedding_worker_failed",
        }
    }
}

/// Variáveis de ambiente que o worker herda. Todo o resto (inclusive JWT_SECRET,
/// PASSWORD_PEPPER, DATABASE_URL e RATE_LIMIT_REDIS_URL) fica de fora.
fn env_allowed(key: &str) -> bool {
    const EXACT: &[&str] = &[
        "PATH", "HOME", "XDG_CACHE_HOME", "TMPDIR", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "RUST_LOG", "RUST_BACKTRACE",
        "SANDBOX", "SANDBOX_READ_PATHS",
    ];
    const PREFIX: &[&str] = &["DOCLING_", "ORT_", "OMP_", "CUDA_", "NVIDIA_"];
    EXACT.contains(&key) || PREFIX.iter().any(|p| key.starts_with(p))
}

pub fn worker_env(vars: impl IntoIterator<Item = (OsString, OsString)>) -> Vec<(OsString, OsString)> {
    vars.into_iter().filter(|(k, _)| k.to_str().is_some_and(env_allowed)).collect()
}

/// Lê até `cap` bytes e descarta o resto (sem parar de ler: um filho que escreve demais não
/// pode travar bloqueado no pipe). Devolve (dados, estourou?).
async fn read_capped(src: Option<impl AsyncRead + Unpin>, cap: usize) -> (Vec<u8>, bool) {
    let Some(mut src) = src else { return (vec![], false) };
    let mut buf = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    let mut overflow = false;
    loop {
        match src.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if buf.len() + n > cap {
                    overflow = true;
                    let room = cap - buf.len();
                    buf.extend_from_slice(&chunk[..room]);
                } else {
                    buf.extend_from_slice(&chunk[..n]);
                }
            }
        }
    }
    (buf, overflow)
}

/// `<binário> <subcomando>` com ambiente zerado (só a lista permitida), pipes e grupo próprio.
fn base_command(cfg: &Config, subcommand: &str) -> Command {
    let mut cmd = Command::new(&cfg.ocr_worker_exe);
    cmd.arg(subcommand)
        .env_clear()
        .envs(worker_env(std::env::vars_os()))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Grupo de processos próprio: o SIGKILL alcança qualquer descendente que segure os pipes.
        .process_group(0)
        .kill_on_drop(true);
    cmd
}

fn command(cfg: &Config, args: &[OsString]) -> Command {
    let mut cmd = base_command(cfg, "ocr-worker");
    cmd.args(args)
        .args(["--max-pages", &cfg.ocr_max_pages.to_string()])
        .args(["--timeout-secs", &cfg.ocr_timeout.as_secs().to_string()])
        .args(["--memory-mb", &cfg.ocr_worker_memory_mb.to_string()]);
    cmd
}

/// SIGKILL no grupo do worker (enquanto ele não foi colhido, o PGID não pode ser reaproveitado).
fn kill_group(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        // SAFETY: kill(2) com PGID negativo e sinal constante.
        unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL) };
    }
    let _ = child.start_kill();
}

// ------------------------------------------------------------------ vigia de memória

const WATCH_EVERY: Duration = Duration::from_millis(10);
/// Acima disto (% do limite do cgroup, só memória não recuperável) o maior worker morre.
const CGROUP_KILL_PCT: u64 = 85;

/// Worker em execução: o vigia só MARCA; quem mata é a tarefa dona do processo (pelo handle do
/// tokio), então não existe o risco de sinalizar um PID já reaproveitado.
struct Watched {
    kill: AtomicBool,
    reason: Mutex<&'static str>,
    /// Teto de RSS deste worker em bytes (0 = só o limite do contêiner).
    max_rss: u64,
}

static RUNNING: Mutex<Option<HashMap<u32, Arc<Watched>>>> = Mutex::new(None);
static WATCHDOG: Once = Once::new();

fn running() -> std::sync::MutexGuard<'static, Option<HashMap<u32, Arc<Watched>>>> {
    RUNNING.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Memória que o worker não devolve sob pressão (anônima + tmpfs), em bytes.
fn worker_rss(pid: u32) -> u64 {
    let Ok(s) = std::fs::read_to_string(format!("/proc/{pid}/status")) else { return 0 };
    s.lines()
        .filter(|l| l.starts_with("RssAnon:") || l.starts_with("RssShmem:"))
        .filter_map(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        .sum::<u64>()
        * 1024
}

/// `(não recuperável, limite)` do cgroup v2 do contêiner: `memory.current` menos o cache de
/// arquivos, mais o tmpfs (que conta como arquivo, mas não pode ser descartado).
pub fn cgroup_unreclaimable() -> Option<(u64, u64)> {
    let read = |f: &str| std::fs::read_to_string(format!("/sys/fs/cgroup/{f}")).ok();
    let limit: u64 = read("memory.max")?.trim().parse().ok()?; // "max" => sem limite
    let current: u64 = read("memory.current")?.trim().parse().ok()?;
    let stat = read("memory.stat")?;
    let field = |k: &str| {
        stat.lines().find_map(|l| l.strip_prefix(k)?.strip_prefix(' ')?.trim().parse::<u64>().ok()).unwrap_or(0)
    };
    Some((current.saturating_sub(field("file")) + field("shmem"), limit))
}

fn watchdog_tick() {
    let guard = running();
    let Some(map) = guard.as_ref().filter(|m| !m.is_empty()) else { return };
    let rss: Vec<(u32, u64)> = map.keys().map(|&pid| (pid, worker_rss(pid))).collect();
    let mark = |pid: u32, reason: &'static str| {
        if let Some(w) = map.get(&pid) {
            *w.reason.lock().unwrap_or_else(PoisonError::into_inner) = reason;
            w.kill.store(true, Ordering::SeqCst);
        }
    };

    for &(pid, r) in &rss {
        if map.get(&pid).is_some_and(|w| w.max_rss > 0 && r > w.max_rss) {
            mark(pid, "teto por worker");
        }
    }
    if let Some((used, limit)) = cgroup_unreclaimable()
        && (used as u128) * 100 >= (limit as u128) * CGROUP_KILL_PCT as u128
        && let Some(&(pid, _)) = rss.iter().max_by_key(|(_, r)| *r)
    {
        mark(pid, "limite do contêiner");
    }
}

/// Thread própria (e não tarefa do tokio): independe de qual runtime chamou primeiro e continua
/// medindo mesmo com o runtime saturado.
fn start_watchdog() {
    WATCHDOG.call_once(|| {
        let spawned = std::thread::Builder::new().name("ocr-memory-watchdog".into()).spawn(|| {
            loop {
                std::thread::sleep(WATCH_EVERY);
                watchdog_tick();
            }
        });
        if let Err(e) = spawned {
            tracing::error!(error = %e, "vigia de memória do OCR não iniciou");
        }
    });
}

/// Roda o worker, com prazo; mata o processo ao estourar. Devolve o stdout se ele saiu com 0.
async fn run(mut cmd: Command, timeout: Duration, max_rss: u64) -> Result<Vec<u8>, WorkerFailure> {
    start_watchdog();
    let mut child = cmd.spawn().map_err(WorkerFailure::Spawn)?;
    let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
    let watched = Arc::new(Watched { kill: AtomicBool::new(false), reason: Mutex::new(""), max_rss });
    let pid = child.id();
    if let Some(pid) = pid {
        running().get_or_insert_with(HashMap::new).insert(pid, watched.clone());
    }
    let unregister = || {
        if let Some(pid) = pid
            && let Some(m) = running().as_mut()
        {
            m.remove(&pid);
        }
    };

    let waiter = async {
        loop {
            tokio::select! {
                st = child.wait() => break st,
                () = tokio::time::sleep(WATCH_EVERY) => {
                    if watched.kill.load(Ordering::SeqCst) {
                        kill_group(&mut child);
                    }
                }
            }
        }
    };
    let waited = tokio::time::timeout(timeout, async {
        tokio::join!(read_capped(stdout, MAX_REPLY_BYTES), read_capped(stderr, MAX_STDERR_BYTES), waiter)
    })
    .await;
    unregister();

    if watched.kill.load(Ordering::SeqCst) {
        kill_group(&mut child);
        let _ = child.wait().await;
        return Err(WorkerFailure::MemoryLimit(*watched.reason.lock().unwrap_or_else(PoisonError::into_inner)));
    }

    let ((out, out_overflow), (err, _), status) = match waited {
        Ok(r) => r,
        Err(_) => {
            kill_group(&mut child);
            let _ = child.wait().await; // só libera a vaga depois que o processo morreu
            return Err(WorkerFailure::Timeout);
        }
    };
    let status = status.map_err(WorkerFailure::Spawn)?;
    if !err.is_empty() {
        tracing::debug!(stderr = %String::from_utf8_lossy(&err), "saída de erro do worker de OCR");
    }
    if !status.success() {
        use std::os::unix::process::ExitStatusExt;
        if !err.is_empty() {
            tracing::warn!(stderr = %String::from_utf8_lossy(&err), "worker de OCR falhou");
        }
        return Err(match status.signal() {
            Some(sig) => WorkerFailure::Signaled(sig),
            None => WorkerFailure::Exited(status.code().unwrap_or(-1)),
        });
    }
    if out_overflow {
        return Err(WorkerFailure::ReplyTooLarge);
    }
    Ok(out)
}

/// Processa `file` (que está dentro de `job_dir`, exclusivo deste job).
pub async fn run_job(cfg: &Config, job_dir: &Path, file: &Path) -> Result<Reply, WorkerFailure> {
    debug_assert!(file.starts_with(job_dir));
    let cmd = command(cfg, &["--file".into(), file.as_os_str().to_owned()]);
    // Margem sobre o prazo do motor: o worker para sozinho entre páginas; o SIGKILL é o último recurso.
    let out = run(cmd, cfg.ocr_timeout + Duration::from_secs(5), cfg.ocr_worker_max_rss_mb * 1024 * 1024).await?;
    serde_json::from_slice(&out).map_err(|_| WorkerFailure::BadReply)
}

/// Divide o texto de `file` (o job, dentro de `job_dir`, exclusivo deste job).
pub async fn run_chunk_job(cfg: &Config, job_dir: &Path, file: &Path) -> Result<chunk::Reply, WorkerFailure> {
    debug_assert!(file.starts_with(job_dir));
    let mut cmd = base_command(cfg, "chunk-worker");
    cmd.arg("--file").arg(file).args(["--timeout-secs", &cfg.chunk.timeout.as_secs().to_string()]);
    if let Some(t) = &cfg.chunk.tokenizer {
        cmd.arg("--tokenizer").arg(t);
    }
    if let Some(d) = &cfg.chunk.nltk_data {
        cmd.arg("--nltk-data").arg(d);
    }
    let out = run(cmd, cfg.chunk.timeout, cfg.chunk.worker_max_rss_mb * 1024 * 1024).await?;
    serde_json::from_slice(&out).map_err(|_| WorkerFailure::BadReply)
}

/// Calcula os embeddings do job `file` (dentro de `job_dir`). `model_dir`: liberado para leitura
/// no Landlock do worker.
pub async fn run_embedding_job(cfg: &Config, job_dir: &Path, file: &Path, model_dir: Option<&Path>) -> Result<crate::embed::Reply, WorkerFailure> {
    debug_assert!(file.starts_with(job_dir));
    let mut cmd = base_command(cfg, "embed-worker");
    cmd.arg("--file").arg(file).args(["--timeout-secs", &cfg.embedding.timeout.as_secs().to_string()]);
    if let Some(d) = model_dir {
        cmd.arg("--model-dir").arg(d);
    }
    let out = run(cmd, cfg.embedding.timeout, cfg.embedding.worker_max_rss_mb * 1024 * 1024).await?;
    serde_json::from_slice(&out).map_err(|_| WorkerFailure::BadReply)
}

/// Pré-carrega os modelos num worker confiável (sem entrada de usuário), o único autorizado a
/// gravar o cache de grafos. Roda em segundo plano na subida do servidor.
pub fn spawn_warmup(cfg: Config) {
    tokio::spawn(async move {
        let started = std::time::Instant::now();
        match run(command(&cfg, &["--warmup".into()]), WARMUP_TIMEOUT, cfg.ocr_worker_max_rss_mb * 1024 * 1024).await {
            Ok(_) => tracing::info!(secs = started.elapsed().as_secs(), "modelos do OCR pré-carregados"),
            Err(e) => tracing::warn!(error = %e, "pré-carga dos modelos do OCR falhou (os jobs carregam sozinhos, mais devagar)"),
        }
    });
}

// ------------------------------------------------------------------ lado do worker

pub struct WorkerArgs {
    pub file: Option<PathBuf>,
    pub warmup: bool,
    pub max_pages: usize,
    pub timeout: Duration,
    pub memory_mb: u64,
}

/// Aplica todo o confinamento do worker. Chamado ANTES de criar qualquer thread. `extra_read`:
/// arquivos que o job precisa ler além do sistema e dos modelos do OCR (ex.: o tokenizador).
pub fn confine(
    job_dir: Option<&Path>,
    graph_dir: Option<&Path>,
    graph_writable: bool,
    extra_read: &[PathBuf],
    limits: &Limits,
) -> anyhow::Result<()> {
    let mode = Mode::from_env().map_err(|e| anyhow::anyhow!(e))?;

    hardening::apply_limits(limits)?;
    // Antes do PR_SET_DUMPABLE: depois dele o /proc/self passa a ser do root.
    hardening::prefer_oom_kill()?;
    hardening::set_not_dumpable()?;

    let plan = sandbox::worker_plan(job_dir, graph_dir, graph_writable, extra_read);
    match sandbox::apply(&plan, mode)? {
        Outcome::Enforced | Outcome::Disabled => {}
        other => tracing::warn!(?other, "worker sem Landlock completo"),
    }
    if let Err(e) = sandbox::apply_scopes(mode) {
        tracing::warn!(error = %e, "Landlock scopes indisponível");
    }
    if mode != Mode::Off
        && let Err(e) = hardening::apply_seccomp(Profile::Worker { allow_network: false }) {
            if mode == Mode::Required {
                return Err(e.into());
            }
            tracing::warn!(error = %e, "seccomp indisponível no worker");
        }
    Ok(())
}

/// O runtime da CUDA reserva espaços de endereçamento enormes: com qualquer teto de `RLIMIT_AS`
/// o cuDNN falha (CUDNN_STATUS_INTERNAL_ERROR). Na GPU o teto de memória fica de fora; CPU,
/// arquivos e descritores continuam limitados.
fn gpu_enabled() -> bool {
    std::env::var("DOCLING_RS_EP").is_ok_and(|v| v.trim().eq_ignore_ascii_case("cuda"))
}

pub fn limits_for(args: &WorkerArgs) -> Limits {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get() as u64);
    Limits {
        address_space: (args.memory_mb > 0 && !gpu_enabled()).then(|| args.memory_mb * 1024 * 1024),
        // Tempo de CPU somando todas as threads: prazo x núcleos, com folga. O prazo de parede é
        // da API (SIGKILL); isto só segura um worker órfão.
        cpu_secs: (args.timeout.as_secs() + 30) * cores,
        file_size: 512 * 1024 * 1024,
        open_files: 1024,
    }
}

/// Monta o cache privado do job: um link para cada grafo confiável. Acertos leem o arquivo
/// confiável (somente leitura); faltas gravam aqui dentro e somem com o diretório do job.
fn private_graph_cache(job_dir: &Path, trusted: Option<&Path>) -> std::io::Result<PathBuf> {
    let private = job_dir.join(PRIVATE_GRAPHS);
    std::fs::create_dir(&private)?;
    if let Some(trusted) = trusted {
        for entry in std::fs::read_dir(trusted)?.flatten() {
            let name = entry.file_name();
            if name.to_str().is_some_and(|n| n.ends_with(".onnx")) {
                std::os::unix::fs::symlink(entry.path(), private.join(&name))?;
            }
        }
    }
    Ok(private)
}

fn write_reply(reply: &impl serde::Serialize) -> ExitCode {
    let Ok(json) = serde_json::to_vec(reply) else { return ExitCode::from(1) };
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    if out.write_all(&json).and_then(|_| out.flush()).is_err() {
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Warm-up: carrega os modelos e converte o documento embutido, gravando o cache confiável.
fn warmup(args: &WorkerArgs, trusted: Option<&Path>) -> anyhow::Result<()> {
    let job = tempfile::Builder::new().prefix("ocr-warmup-").tempdir()?;
    let doc = job.path().join("warmup.pdf");
    std::fs::write(&doc, WARMUP_DOC)?;
    confine(Some(job.path()), trusted, true, &[], &limits_for(args))?;
    data_preparation::warm_up_models().map_err(|e| anyhow::anyhow!(e))?;
    match ocr::execute_job(&doc, args.max_pages, args.timeout).map_err(|e| anyhow::anyhow!(e))? {
        Reply::Ok { .. } => Ok(()),
        Reply::Rejected { reason } => anyhow::bail!("documento de warm-up recusado: {reason:?}"),
    }
}

/// Ponto de entrada do subcomando `ocr-worker`. Roda numa única thread até o confinamento.
pub fn main(args: WorkerArgs) -> ExitCode {
    let trusted = sandbox::trusted_graph_dir();

    if args.warmup {
        return match warmup(&args, trusted.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("{e}");
                ExitCode::from(1)
            }
        };
    }

    let Some(file) = args.file.as_deref() else {
        eprintln!("--file ausente");
        return ExitCode::from(2);
    };
    let Some(job_dir) = file.parent() else {
        eprintln!("--file sem diretório");
        return ExitCode::from(2);
    };
    match private_graph_cache(job_dir, trusted.as_deref()) {
        // SAFETY: ainda não há outras threads neste processo.
        Ok(private) => unsafe { std::env::set_var("DOCLING_RS_GRAPH_CACHE_DIR", private) },
        Err(e) => {
            eprintln!("cache privado de grafos: {e}");
            // SAFETY: idem.
            unsafe { std::env::set_var("DOCLING_RS_NO_GRAPH_CACHE", "1") };
        }
    }
    let t = std::time::Instant::now();
    if let Err(e) = confine(Some(job_dir), trusted.as_deref(), false, &[], &limits_for(&args)) {
        eprintln!("confinamento do worker falhou: {e}");
        return ExitCode::from(3);
    }
    let confine_ms = t.elapsed().as_millis() as u64;

    match ocr::execute_job(file, args.max_pages, args.timeout) {
        Ok(mut reply) => {
            if let Reply::Ok { timings, .. } = &mut reply {
                timings.confine_ms = confine_ms;
            }
            write_reply(&reply)
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

// ------------------------------------------------------------------ worker de chunking

pub struct ChunkWorkerArgs {
    pub file: PathBuf,
    pub timeout: Duration,
    pub tokenizer: Option<PathBuf>,
    pub nltk_data: Option<PathBuf>,
}

/// Ponto de entrada do subcomando `chunk-worker`. Confina antes de ler qualquer coisa do job.
/// Sem cache de grafos nem ONNX: só texto, tokenizador e dados do NLTK.
pub fn chunk_main(args: ChunkWorkerArgs) -> ExitCode {
    let Some(job_dir) = args.file.parent() else {
        eprintln!("--file sem diretório");
        return ExitCode::from(2);
    };
    let extra: Vec<PathBuf> = args.tokenizer.iter().chain(args.nltk_data.iter()).cloned().collect();
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get() as u64);
    let limits = Limits {
        address_space: None,
        cpu_secs: (args.timeout.as_secs() + 30) * cores,
        // O worker não grava arquivos (a resposta sai pelo pipe).
        file_size: 16 * 1024 * 1024,
        open_files: 256,
    };
    let t = std::time::Instant::now();
    if let Err(e) = confine(Some(job_dir), None, false, &extra, &limits) {
        eprintln!("confinamento do worker falhou: {e}");
        return ExitCode::from(3);
    }
    let confine_ms = t.elapsed().as_millis() as u64;

    match chunk::execute_job(&args.file, args.tokenizer.as_deref(), args.nltk_data.as_deref()) {
        Ok(mut reply) => {
            if let chunk::Reply::Ok { timings, .. } = &mut reply {
                timings.confine_ms = confine_ms;
            }
            write_reply(&reply)
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

// ------------------------------------------------------------------ worker de embeddings

pub struct EmbedWorkerArgs {
    pub file: PathBuf,
    pub timeout: Duration,
    pub model_dir: Option<PathBuf>,
}

/// Ponto de entrada do subcomando `embed-worker`. Confina antes de ler o job; o modelo é lido
/// só do diretório local configurado (sem rede: nada é baixado).
pub fn embed_main(args: EmbedWorkerArgs) -> ExitCode {
    let Some(job_dir) = args.file.parent() else {
        eprintln!("--file sem diretório");
        return ExitCode::from(2);
    };
    let extra: Vec<PathBuf> = args.model_dir.as_deref().map(crate::embed::model_read_paths).unwrap_or_default();
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get() as u64);
    let limits = Limits {
        address_space: None,
        cpu_secs: (args.timeout.as_secs() + 30) * cores,
        // O worker não grava arquivos (a resposta sai pelo pipe).
        file_size: 16 * 1024 * 1024,
        open_files: 1024,
    };
    let t = std::time::Instant::now();
    if let Err(e) = confine(Some(job_dir), None, false, &extra, &limits) {
        eprintln!("confinamento do worker falhou: {e}");
        return ExitCode::from(3);
    }
    let confine_ms = t.elapsed().as_millis() as u64;
    match crate::embed::execute_job(&args.file) {
        Ok(mut reply) => {
            if let crate::embed::Reply::Ok { timings, .. } = &mut reply {
                timings.confine_ms = confine_ms;
            }
            write_reply(&reply)
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_never_reach_the_worker() {
        let vars = [
            ("JWT_SECRET", "x"),
            ("JWT_SECRET_PREVIOUS", "x"),
            ("PASSWORD_PEPPER", "x"),
            ("DATABASE_URL", "postgres://u:senha@db/x"),
            ("RATE_LIMIT_REDIS_URL", "redis://:senha@r"),
            ("AWS_SECRET_ACCESS_KEY", "x"),
            ("PATH", "/usr/bin"),
            ("DOCLING_RS_EP", "cpu"),
            ("SANDBOX", "required"),
        ]
        .map(|(k, v)| (OsString::from(k), OsString::from(v)));
        let kept: Vec<String> = worker_env(vars).into_iter().map(|(k, _)| k.into_string().unwrap()).collect();
        assert_eq!(kept, ["PATH", "DOCLING_RS_EP", "SANDBOX"]);
    }

    #[tokio::test]
    async fn read_capped_drains_past_the_cap() {
        let data = vec![1u8; 300 * 1024];
        let (buf, overflow) = read_capped(Some(&data[..]), 1000).await;
        assert_eq!(buf.len(), 1000);
        assert!(overflow);
        let (buf, overflow) = read_capped(Some(&data[..10]), 1000).await;
        assert_eq!(buf.len(), 10);
        assert!(!overflow);
    }
}
