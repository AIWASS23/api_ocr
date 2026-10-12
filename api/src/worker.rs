//! Worker de OCR: cada documento é inspecionado e convertido num PROCESSO separado.
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
//! * `rlimit` de memória (espaço de endereçamento), CPU, tamanho de arquivo e descritores;
//! * não-depurável e sem core dump (um core teria o conteúdo do documento);
//! * um processo novo por documento: um job comprometido não contamina o próximo;
//! * prazo estourado ou cliente desconectado => SIGKILL (antes, a thread seguia rodando).
//!
//! Protocolo: a API executa `<binário> ocr-worker --file <caminho> ...`; o worker escreve UMA
//! resposta JSON (`ocr::Reply`) no stdout. O stdout e o stderr são lidos com teto de tamanho.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{ExitCode, Stdio};
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

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
}

impl WorkerFailure {
    /// Nome do evento de segurança. Morte por sinal (SIGSYS do seccomp, SIGSEGV, SIGKILL do
    /// limite de CPU...) é o que se espera ver quando um arquivo tenta explorar o parser.
    pub fn event_kind(&self) -> &'static str {
        match self {
            WorkerFailure::Signaled(_) => "ocr_worker_crashed",
            WorkerFailure::ReplyTooLarge | WorkerFailure::BadReply => "ocr_worker_bad_reply",
            _ => "ocr_worker_failed",
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

fn command(cfg: &Config, args: &[OsString]) -> Command {
    let mut cmd = Command::new(&cfg.ocr_worker_exe);
    cmd.arg("ocr-worker")
        .args(args)
        .args(["--max-pages", &cfg.ocr_max_pages.to_string()])
        .args(["--timeout-secs", &cfg.ocr_timeout.as_secs().to_string()])
        .args(["--memory-mb", &cfg.ocr_worker_memory_mb.to_string()])
        .env_clear()
        .envs(worker_env(std::env::vars_os()))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd
}

/// Roda o worker, com prazo; mata o processo ao estourar. Devolve o stdout se ele saiu com 0.
async fn run(mut cmd: Command, timeout: Duration) -> Result<Vec<u8>, WorkerFailure> {
    let mut child = cmd.spawn().map_err(WorkerFailure::Spawn)?;
    let (stdout, stderr) = (child.stdout.take(), child.stderr.take());

    let waited = tokio::time::timeout(timeout, async {
        tokio::join!(read_capped(stdout, MAX_REPLY_BYTES), read_capped(stderr, MAX_STDERR_BYTES), child.wait())
    })
    .await;

    let ((out, out_overflow), (err, _), status) = match waited {
        Ok(r) => r,
        Err(_) => {
            let _ = child.start_kill();
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
    let out = run(cmd, cfg.ocr_timeout + Duration::from_secs(5)).await?;
    serde_json::from_slice(&out).map_err(|_| WorkerFailure::BadReply)
}

/// Pré-carrega os modelos num worker confiável (sem entrada de usuário), o único autorizado a
/// gravar o cache de grafos. Roda em segundo plano na subida do servidor.
pub fn spawn_warmup(cfg: Config) {
    tokio::spawn(async move {
        let started = std::time::Instant::now();
        match run(command(&cfg, &["--warmup".into()]), WARMUP_TIMEOUT).await {
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

/// Aplica todo o confinamento do worker. Chamado ANTES de criar qualquer thread.
pub fn confine(job_dir: Option<&Path>, graph_dir: Option<&Path>, graph_writable: bool, limits: &Limits) -> anyhow::Result<()> {
    let mode = Mode::from_env().map_err(|e| anyhow::anyhow!(e))?;

    hardening::apply_limits(limits)?;
    hardening::set_not_dumpable()?;

    let plan = sandbox::worker_plan(job_dir, graph_dir, graph_writable);
    match sandbox::apply(&plan, mode)? {
        Outcome::Enforced | Outcome::Disabled => {}
        other => tracing::warn!(?other, "worker de OCR sem Landlock completo"),
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

fn write_reply(reply: &Reply) -> ExitCode {
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
    confine(Some(job.path()), trusted, true, &limits_for(args))?;
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
    if let Err(e) = confine(Some(job_dir), trusted.as_deref(), false, &limits_for(&args)) {
        eprintln!("confinamento do worker falhou: {e}");
        return ExitCode::from(3);
    }

    match ocr::execute_job(file, args.max_pages, args.timeout) {
        Ok(reply) => write_reply(&reply),
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
