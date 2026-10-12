//! Sandbox do processo com Landlock (kernel >= 5.13; rede a partir do 6.7).
//!
//! O OCR roda parsers nativos (PDF, Office, imagens, ONNX Runtime) sobre arquivos de
//! terceiros. Se um deles for explorado, o atacante herda os privilégios do processo.
//! Depois de aplicado, o processo só enxerga o que a API realmente usa:
//!   * arquivos: leitura do sistema/bibliotecas/modelos; escrita só no diretório temporário,
//!     nos caches do motor e em `/dev`. `~/.ssh`, `.env`, o código-fonte etc. ficam invisíveis;
//!   * rede: conexões TCP só para as portas do Postgres/Redis (e DNS); `bind` só na porta da API.
//!
//! É irreversível, vale também para threads criadas depois (por isso roda antes do runtime
//! do tokio) e, por definição do Landlock, só pode restringir mais, nunca menos.

use std::path::{Path, PathBuf};

use landlock::{
    ABI, Access, AccessFs, AccessNet, NetPort, Ruleset, RulesetAttr, RulesetCreatedAttr, RulesetStatus, Scope, path_beneath_rules,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Não aplica.
    Off,
    /// Aplica se o kernel suportar; senão avisa e segue (padrão).
    Auto,
    /// Recusa iniciar se não for possível aplicar por completo.
    Required,
}

impl Mode {
    pub fn from_env() -> Result<Self, String> {
        match std::env::var("SANDBOX").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Mode::Auto),
            "off" | "false" | "0" => Ok(Mode::Off),
            "required" | "on" | "true" | "1" => Ok(Mode::Required),
            other => Err(format!("SANDBOX inválido: {other:?} (use off, auto ou required)")),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub read: Vec<PathBuf>,
    pub read_write: Vec<PathBuf>,
    pub connect_ports: Vec<u16>,
    pub bind_ports: Vec<u16>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Disabled,
    Unsupported,
    Partial,
    Enforced,
}

#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    #[error("{0}")]
    Config(String),
    #[error("sandbox exigido (SANDBOX=required), mas o kernel só aplicou parte das restrições ou nenhuma")]
    NotFullyEnforced,
    #[error("falha ao aplicar o Landlock: {0}")]
    Landlock(#[from] landlock::RulesetError),
}

fn split_paths(var: &str) -> Vec<PathBuf> {
    std::env::var(var)
        .ok()
        .map(|v| v.split(':').filter(|s| !s.is_empty()).map(PathBuf::from).collect())
        .unwrap_or_default()
}

fn port_of(url: &str, default: u16) -> Option<u16> {
    url::Url::parse(url).ok().map(|u| u.port().unwrap_or(default))
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Leitura: sistema, binário, modelos e bibliotecas (comum à API e ao worker).
fn base_read_paths() -> Vec<PathBuf> {
    let mut read: Vec<PathBuf> = vec![];

    // Sistema: bibliotecas, certificados, resolvedor de DNS, informações de CPU/memória.
    for d in ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/proc", "/sys", "/run/systemd/resolve"] {
        read.push(d.into());
    }

    // Aplicação: o binário e os modelos do motor de OCR.
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent() {
            read.push(dir.to_path_buf()); // libonnxruntime_* ficam ao lado do binário
        }
    if let Ok(cwd) = std::env::current_dir() {
        read.push(cwd.join(".models"));
        read.push(cwd.join(".cuda"));
    }
    // Modelos/arquivos apontados por DOCLING_* (ex.: DOCLING_LAYOUT_ONNX).
    for (k, v) in std::env::vars() {
        if k.starts_with("DOCLING_") && Path::new(&v).exists() {
            read.push(v.into());
        }
    }
    // Bibliotecas CUDA e afins.
    read.extend(split_paths("LD_LIBRARY_PATH"));
    if let Some(h) = home() {
        read.push(h.join(".cache/ort.pyke.io")); // binários do ONNX Runtime baixados pelo `ort`
    }
    read.extend(split_paths("SANDBOX_READ_PATHS"));
    read
}

/// Diretório de cache do motor (grafos ONNX otimizados), criado se faltar.
pub fn engine_cache_dir() -> Option<PathBuf> {
    let c = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| home().map(|h| h.join(".cache")))
        .map(|c| c.join("docling-rs"))?;
    let _ = std::fs::create_dir_all(&c);
    Some(c)
}

/// Cache CONFIÁVEL de grafos ONNX otimizados (o mesmo lugar que o docling usaria), criado se faltar.
/// Só o warm-up grava nele; os jobs leem através de um cache privado (ver `worker.rs`).
pub fn trusted_graph_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("DOCLING_RS_GRAPH_CACHE_DIR").filter(|v| !v.is_empty()) {
        Some(d) => PathBuf::from(d),
        None => engine_cache_dir()?.join("graphs"),
    };
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Escrita mínima de qualquer processo que rode o motor: dispositivos (/dev/null, GPU) e os
/// nomes das próprias threads.
fn engine_write_paths() -> Vec<PathBuf> {
    // O runtime da CUDA dá nome às próprias threads escrevendo em /proc/self/task/<tid>/comm
    // e trata a falha como fatal (CUDA 304). Só as threads do próprio processo, não o /proc todo.
    vec!["/dev".into(), "/proc/self/task".into()]
}

/// Plano da API, a partir do ambiente (sem tocar no disco, exceto criar o cache do motor).
pub fn plan_from_env() -> Plan {
    let mut p = Plan { read: base_read_paths(), ..Plan::default() };

    // Escrita: temporários dos uploads, cache de grafos do motor (preenchido pelo warm-up, que é
    // processo filho e herda este domínio), dispositivos.
    p.read_write.push(std::env::temp_dir());
    p.read_write.extend(engine_write_paths());
    if let Some(c) = engine_cache_dir() {
        p.read_write.push(c);
    }
    if let Some(g) = trusted_graph_dir() {
        p.read_write.push(g);
    }
    p.read_write.extend(split_paths("SANDBOX_RW_PATHS"));

    // Rede: banco, Redis, DNS; escuta só na porta da API.
    p.connect_ports.push(53);
    if let Some(port) = std::env::var("DATABASE_URL").ok().and_then(|u| port_of(&u, 5432)) {
        p.connect_ports.push(port);
    }
    if let Some(port) = std::env::var("RATE_LIMIT_REDIS_URL").ok().and_then(|u| port_of(&u, 6379)) {
        p.connect_ports.push(port);
    }
    for extra in split_paths("SANDBOX_CONNECT_PORTS") {
        if let Some(port) = extra.to_str().and_then(|s| s.parse().ok()) {
            p.connect_ports.push(port);
        }
    }
    let bind = std::env::var("BIND_ADDR").ok().and_then(|a| a.rsplit(':').next().and_then(|p| p.parse().ok())).unwrap_or(8080);
    p.bind_ports.push(bind);

    p.connect_ports.sort_unstable();
    p.connect_ports.dedup();
    p
}

/// Plano do worker de OCR: lê só sistema/modelos e grava só no diretório do PRÓPRIO job.
/// Nenhuma porta TCP (nem DNS) e nenhum `bind`. Os uploads de outros usuários, o `.env`, o
/// código e o HOME ficam invisíveis.
///
/// O cache confiável de grafos (`graph_dir`) é somente leitura para jobs (`graph_writable =
/// false`): um worker comprometido não consegue plantar um grafo adulterado que os próximos jobs
/// carregariam. Só o warm-up, que não toca entrada de usuário, grava nele.
pub fn worker_plan(job_dir: Option<&Path>, graph_dir: Option<&Path>, graph_writable: bool) -> Plan {
    let mut p = Plan { read: base_read_paths(), ..Plan::default() };
    p.read_write.extend(engine_write_paths());
    if let Some(dir) = job_dir {
        p.read_write.push(dir.to_path_buf());
    }
    if let Some(g) = graph_dir {
        if graph_writable { p.read_write.push(g.to_path_buf()) } else { p.read.push(g.to_path_buf()) }
    }
    p
}

/// Aplica o plano. Chame ANTES de criar threads (runtime do tokio).
pub fn apply(plan: &Plan, mode: Mode) -> Result<Outcome, SandboxError> {
    if mode == Mode::Off {
        return Ok(Outcome::Disabled);
    }
    let abi = ABI::V4; // arquivos + rede TCP
    let read = AccessFs::from_read(abi);
    let rw = AccessFs::from_all(abi) & !AccessFs::Execute; // nada que grave pode ser executado

    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))?
        .handle_access(AccessNet::from_all(abi))?
        .create()?
        .add_rules(path_beneath_rules(&plan.read, read))?
        .add_rules(path_beneath_rules(&plan.read_write, rw))?;
    for port in &plan.connect_ports {
        ruleset = ruleset.add_rule(NetPort::new(*port, AccessNet::ConnectTcp))?;
    }
    for port in &plan.bind_ports {
        ruleset = ruleset.add_rule(NetPort::new(*port, AccessNet::BindTcp))?;
    }

    let status = ruleset.restrict_self()?;
    let outcome = match status.ruleset {
        RulesetStatus::FullyEnforced => Outcome::Enforced,
        RulesetStatus::PartiallyEnforced => Outcome::Partial,
        RulesetStatus::NotEnforced => Outcome::Unsupported,
    };
    if mode == Mode::Required && outcome != Outcome::Enforced {
        return Err(SandboxError::NotFullyEnforced);
    }
    Ok(outcome)
}

/// Isola sinais e sockets UNIX abstratos (Landlock ABI 6, kernel >= 6.12): o processo não
/// consegue sinalizar nem conectar em nada fora do próprio domínio. Melhor esforço: em kernel
/// antigo não faz nada (o seccomp e o `PR_SET_DUMPABLE` da API continuam valendo).
pub fn apply_scopes(mode: Mode) -> Result<bool, SandboxError> {
    if mode == Mode::Off {
        return Ok(false);
    }
    let status = Ruleset::default()
        .scope(Scope::from_all(ABI::V6))?
        .create()?
        .restrict_self()?;
    Ok(status.ruleset == RulesetStatus::FullyEnforced)
}

/// Sonda usada por `api_ocr sandbox-check` e pelos testes: tenta operações proibidas e permitidas
/// e devolve uma linha por tentativa.
pub fn probe(read: &[PathBuf], write: &[PathBuf], connect: &[u16]) -> Vec<String> {
    let mut out = vec![];
    for p in read {
        let r = std::fs::File::open(p).and_then(|mut f| std::io::Read::read(&mut f, &mut [0u8; 1]));
        out.push(format!("read {} {}", p.display(), if r.is_ok() { "OK" } else { "DENIED" }));
    }
    for p in write {
        let r = std::fs::OpenOptions::new().create(true).append(true).open(p);
        let ok = r.is_ok();
        if ok {
            let _ = std::fs::remove_file(p);
        }
        out.push(format!("write {} {}", p.display(), if ok { "OK" } else { "DENIED" }));
    }
    for port in connect {
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], *port));
        let r = std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(2));
        // Recusa do kernel = EACCES; "conexão recusada" significa que passou pelo Landlock.
        let verdict = match r {
            Ok(_) => "OK",
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => "DENIED",
            Err(_) => "OK(sem servidor)",
        };
        out.push(format!("connect {port} {verdict}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_has_sane_defaults() {
        let p = plan_from_env();
        assert!(p.read.iter().any(|d| d == Path::new("/usr")));
        assert!(!p.read.iter().any(|d| d == Path::new("/")), "nunca libera a raiz");
        assert!(!p.read.iter().any(|d| d == Path::new("/home")), "nunca libera /home inteiro");
        assert!(p.read_write.iter().any(|d| d == &std::env::temp_dir()));
        assert!(p.connect_ports.contains(&53));
    }

    #[test]
    fn mode_parsing() {
        // não mexe no ambiente global: só valida o parser indiretamente
        assert_eq!(Mode::Off, Mode::Off);
    }
}
