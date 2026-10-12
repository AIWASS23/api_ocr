//! Endurecimento do processo além do Landlock: seccomp, limites de recursos e atributos do processo.
//!
//! O Landlock decide QUAIS arquivos e portas o processo alcança; o seccomp decide QUAIS chamadas de
//! sistema ele pode fazer. Os filtros aqui são listas de negação: bloqueiam o que um invasor usaria
//! para escapar ou espionar (ptrace, ler memória de outro processo, montar sistemas de arquivos,
//! carregar módulo, criar namespaces, eBPF, io_uring...) sem arriscar quebrar o ONNX Runtime ou as
//! bibliotecas de PDF/imagem com uma lista de permitidos incompleta. O worker de OCR recebe, além
//! disso, a proibição de executar programas e de abrir sockets de rede.
//!
//! Filtros seccomp são herdados por threads e processos filhos e sobrevivem ao `execve`: o worker
//! herda o da API e acrescenta o seu.

use std::collections::BTreeMap;

use seccompiler::{
    BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule, TargetArch,
};

#[derive(Debug, thiserror::Error)]
pub enum HardeningError {
    #[error("arquitetura sem suporte a seccomp: {0}")]
    UnsupportedArch(&'static str),
    #[error("falha ao montar o filtro seccomp: {0}")]
    Filter(String),
    #[error("falha ao aplicar o filtro seccomp: {0}")]
    Apply(String),
    #[error("falha em {0}: {1}")]
    Sys(&'static str, std::io::Error),
}

/// Perfil de chamadas de sistema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Processo da API: precisa de rede e de executar o worker.
    Api,
    /// Worker de OCR: sem `execve` e, salvo `allow_network`, sem sockets de rede.
    Worker { allow_network: bool },
}

const EPERM: u32 = libc::EPERM as u32;
const ENOSYS: u32 = libc::ENOSYS as u32;

/// Namespaces: criar um (inclusive de usuário) é o primeiro passo de muitos escapes.
const NS_FLAGS: u64 = (libc::CLONE_NEWUSER
    | libc::CLONE_NEWNS
    | libc::CLONE_NEWNET
    | libc::CLONE_NEWPID
    | libc::CLONE_NEWIPC
    | libc::CLONE_NEWUTS
    | libc::CLONE_NEWCGROUP) as u64;

/// Chamadas negadas para todos os perfis.
fn denied_everywhere() -> Vec<i64> {
    let mut v = vec![
        // Espionar/alterar outros processos.
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_pidfd_getfd,
        libc::SYS_kcmp,
        // Kernel, módulos, eBPF, perf.
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_kexec_load,
        libc::SYS_kexec_file_load,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_userfaultfd,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        // Sistema de arquivos e namespaces.
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_chroot,
        libc::SYS_unshare,
        libc::SYS_setns,
        libc::SYS_open_tree,
        libc::SYS_move_mount,
        libc::SYS_fsopen,
        libc::SYS_fsconfig,
        libc::SYS_fsmount,
        libc::SYS_fspick,
        libc::SYS_mount_setattr,
        libc::SYS_open_by_handle_at,
        libc::SYS_name_to_handle_at,
        libc::SYS_fanotify_init,
        // Administração da máquina.
        libc::SYS_swapon,
        libc::SYS_swapoff,
        libc::SYS_reboot,
        libc::SYS_acct,
        libc::SYS_settimeofday,
        libc::SYS_clock_settime,
        libc::SYS_clock_adjtime,
        libc::SYS_adjtimex,
        libc::SYS_quotactl,
        libc::SYS_sethostname,
        libc::SYS_setdomainname,
        libc::SYS_syslog,
        libc::SYS_vhangup,
    ];
    #[cfg(target_arch = "x86_64")]
    v.extend([libc::SYS_iopl, libc::SYS_ioperm, libc::SYS_modify_ldt, libc::SYS_uselib, libc::SYS_lookup_dcookie]);
    v
}

fn target_arch() -> Result<TargetArch, HardeningError> {
    TargetArch::try_from(std::env::consts::ARCH).map_err(|_| HardeningError::UnsupportedArch(std::env::consts::ARCH))
}

fn cond(arg: u8, op: SeccompCmpOp, value: u64) -> Result<SeccompCondition, HardeningError> {
    SeccompCondition::new(arg, SeccompCmpArgLen::Qword, op, value).map_err(|e| HardeningError::Filter(e.to_string()))
}

fn rule(conds: Vec<SeccompCondition>) -> Result<SeccompRule, HardeningError> {
    SeccompRule::new(conds).map_err(|e| HardeningError::Filter(e.to_string()))
}

fn compile(rules: BTreeMap<i64, Vec<SeccompRule>>, action: SeccompAction) -> Result<BpfProgram, HardeningError> {
    let filter = SeccompFilter::new(rules, SeccompAction::Allow, action, target_arch()?)
        .map_err(|e| HardeningError::Filter(e.to_string()))?;
    BpfProgram::try_from(filter).map_err(|e| HardeningError::Filter(e.to_string()))
}

/// Monta os programas BPF do perfil (mais de um porque cada um tem a sua ação de recusa).
pub fn filters(profile: Profile) -> Result<Vec<BpfProgram>, HardeningError> {
    let mut deny: BTreeMap<i64, Vec<SeccompRule>> = denied_everywhere().into_iter().map(|s| (s, vec![])).collect();

    // clone(2) com flags de namespace. Uma regra por flag: basta qualquer uma estar ligada.
    let mut ns_rules = vec![];
    for bit in 0..64 {
        let flag = 1u64 << bit;
        if NS_FLAGS & flag != 0 {
            ns_rules.push(rule(vec![cond(0, SeccompCmpOp::MaskedEq(flag), flag)?])?);
        }
    }
    deny.insert(libc::SYS_clone, ns_rules);

    if let Profile::Worker { allow_network } = profile {
        deny.insert(libc::SYS_execve, vec![]);
        deny.insert(libc::SYS_execveat, vec![]);
        if !allow_network {
            // Só sockets locais (AF_UNIX) continuam possíveis; nada de IPv4/IPv6/netlink/raw.
            deny.insert(libc::SYS_socket, vec![rule(vec![cond(0, SeccompCmpOp::Ne, libc::AF_UNIX as u64)?])?]);
        }
    }

    // clone3 recebe as flags numa struct que o BPF não lê: ENOSYS faz a libc cair no clone(2),
    // que é filtrado acima (o mesmo que Docker e systemd fazem).
    let clone3 = BTreeMap::from([(libc::SYS_clone3, vec![])]);

    Ok(vec![compile(deny, SeccompAction::Errno(EPERM))?, compile(clone3, SeccompAction::Errno(ENOSYS))?])
}

/// `PR_SET_NO_NEW_PRIVS`: nem setuid nem capacidades de arquivo depois daqui (exigido pelo seccomp).
pub fn no_new_privs() -> Result<(), HardeningError> {
    // SAFETY: prctl com argumentos inteiros, sem ponteiros.
    let r = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if r != 0 {
        return Err(HardeningError::Sys("PR_SET_NO_NEW_PRIVS", std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Aplica o perfil à thread atual (e a tudo que ela criar depois). Chame antes de criar threads.
pub fn apply_seccomp(profile: Profile) -> Result<(), HardeningError> {
    no_new_privs()?;
    for prog in filters(profile)? {
        seccompiler::apply_filter(&prog).map_err(|e| HardeningError::Apply(e.to_string()))?;
    }
    Ok(())
}

/// `PR_SET_DUMPABLE=0`: `/proc/<pid>/mem`, `environ`, `maps` etc. passam a pertencer ao root e
/// o processo não pode ser depurado nem gera core dump. Protege os segredos da API contra outro
/// processo do mesmo usuário (inclusive um worker comprometido).
pub fn set_not_dumpable() -> Result<(), HardeningError> {
    // SAFETY: prctl com argumentos inteiros, sem ponteiros.
    let r = unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) };
    if r != 0 {
        return Err(HardeningError::Sys("PR_SET_DUMPABLE", std::io::Error::last_os_error()));
    }
    Ok(())
}

pub fn is_dumpable() -> bool {
    // SAFETY: prctl de leitura, sem ponteiros.
    unsafe { libc::prctl(libc::PR_GET_DUMPABLE, 0, 0, 0, 0) == 1 }
}

/// Limites de recursos do worker de OCR.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Espaço de endereçamento (bytes). `None` = sem limite (necessário com CUDA).
    pub address_space: Option<u64>,
    /// Tempo de CPU (segundos, somando todas as threads).
    pub cpu_secs: u64,
    /// Maior arquivo que o processo pode gravar (bytes).
    pub file_size: u64,
    /// Descritores de arquivo abertos.
    pub open_files: u64,
}

fn setrlimit(resource: libc::__rlimit_resource_t, value: u64, name: &'static str) -> Result<(), HardeningError> {
    let lim = libc::rlimit { rlim_cur: value as libc::rlim_t, rlim_max: value as libc::rlim_t };
    // SAFETY: ponteiro para struct local válida durante a chamada.
    let r = unsafe { libc::setrlimit(resource, &lim) };
    if r != 0 {
        return Err(HardeningError::Sys(name, std::io::Error::last_os_error()));
    }
    Ok(())
}

pub fn apply_limits(l: &Limits) -> Result<(), HardeningError> {
    setrlimit(libc::RLIMIT_CORE, 0, "RLIMIT_CORE")?;
    setrlimit(libc::RLIMIT_CPU, l.cpu_secs.max(1), "RLIMIT_CPU")?;
    setrlimit(libc::RLIMIT_FSIZE, l.file_size, "RLIMIT_FSIZE")?;
    setrlimit(libc::RLIMIT_NOFILE, l.open_files, "RLIMIT_NOFILE")?;
    if let Some(bytes) = l.address_space {
        setrlimit(libc::RLIMIT_AS, bytes, "RLIMIT_AS")?;
    }
    Ok(())
}

/// Resultado de uma tentativa feita pela sonda de diagnóstico (`api_ocr sandbox-check`).
pub fn probe_syscalls() -> Vec<String> {
    let mut out = vec![];
    let verdict = |ok: bool| if ok { "OK" } else { "DENIED" };

    // execve de um binário inofensivo.
    let exec = std::process::Command::new("/bin/true").status().is_ok();
    out.push(format!("syscall execve {}", verdict(exec)));

    // socket IPv4.
    let inet = std::net::UdpSocket::bind("127.0.0.1:0").is_ok();
    out.push(format!("syscall socket-inet {}", verdict(inet)));

    // Ler memória de processo (aqui, a própria: inofensivo, mas o filtro nega a chamada em si).
    let src = [7u8; 8];
    let mut dst = [0u8; 8];
    let local = libc::iovec { iov_base: dst.as_mut_ptr().cast(), iov_len: dst.len() };
    let remote = libc::iovec { iov_base: src.as_ptr() as *mut libc::c_void, iov_len: src.len() };
    // SAFETY: os dois iovec apontam para buffers locais válidos do tamanho declarado.
    let n = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
    out.push(format!("syscall process_vm_readv {}", verdict(n == 8)));

    // unshare de namespace de usuário.
    // SAFETY: flags inteiras.
    let unshare = unsafe { libc::unshare(libc::CLONE_NEWUSER) } == 0;
    out.push(format!("syscall unshare {}", verdict(unshare)));

    // Limites efetivos.
    for (name, res) in [("as", libc::RLIMIT_AS), ("cpu", libc::RLIMIT_CPU), ("nofile", libc::RLIMIT_NOFILE), ("fsize", libc::RLIMIT_FSIZE), ("core", libc::RLIMIT_CORE)] {
        let mut l = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        // SAFETY: ponteiro para struct local.
        unsafe { libc::getrlimit(res, &mut l) };
        let v = if l.rlim_cur == libc::RLIM_INFINITY { "unlimited".to_string() } else { l.rlim_cur.to_string() };
        out.push(format!("rlimit {name} {v}"));
    }
    out.push(format!("dumpable {}", is_dumpable()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_compile_for_every_profile() {
        for p in [Profile::Api, Profile::Worker { allow_network: false }, Profile::Worker { allow_network: true }] {
            let progs = filters(p).expect("compila");
            assert_eq!(progs.len(), 2);
            assert!(progs.iter().all(|p| !p.is_empty()));
        }
    }

    #[test]
    fn worker_filter_is_stricter_than_api() {
        let api = filters(Profile::Api).unwrap()[0].len();
        let worker = filters(Profile::Worker { allow_network: false }).unwrap()[0].len();
        assert!(worker > api);
    }
}
