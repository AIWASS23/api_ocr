//! Confinamento real dos processos: roda `api_ocr sandbox-check` com o perfil do worker de OCR e
//! com o da API, tenta operações proibidas e permitidas e confere o veredito de cada uma.

use std::io::Write;
use std::net::TcpListener;
use std::process::Command;

struct Setup {
    /// "Segredo" fora de qualquer área permitida.
    secret: tempfile::NamedTempFile,
    /// Diretório do job (o único gravável pelo worker).
    job: tempfile::TempDir,
    /// Upload de OUTRO job, no mesmo diretório temporário.
    other_job: tempfile::TempDir,
    db_listener: TcpListener,
}

fn setup() -> Setup {
    let mut secret = tempfile::NamedTempFile::new_in(env!("CARGO_MANIFEST_DIR")).unwrap();
    writeln!(secret, "JWT_SECRET=super-secreto").unwrap();
    let other_job = tempfile::Builder::new().prefix("ocr-").tempdir().unwrap();
    std::fs::write(other_job.path().join("contrato.pdf"), b"%PDF-1.4 documento de outro usuario").unwrap();
    Setup {
        secret,
        job: tempfile::Builder::new().prefix("ocr-").tempdir().unwrap(),
        other_job,
        db_listener: TcpListener::bind("127.0.0.1:0").unwrap(),
    }
}

fn run(profile: &str, mode: &str, s: &Setup) -> (bool, Vec<String>, String) {
    let db_port = s.db_listener.local_addr().unwrap().port();
    let parent_environ = format!("/proc/{}/environ", std::process::id());
    let out = Command::new(env!("CARGO_BIN_EXE_api_ocr"))
        .arg("sandbox-check")
        .args(["--profile", profile])
        .args(["--job-dir", s.job.path().to_str().unwrap()])
        .args(["--read", s.secret.path().to_str().unwrap()])
        .args(["--read", s.other_job.path().join("contrato.pdf").to_str().unwrap()])
        .args(["--read", &parent_environ])
        .args(["--read", "/etc/os-release"])
        .args(["--write", s.job.path().join("saida.tmp").to_str().unwrap()])
        .args(["--write", s.other_job.path().join("plantado.tmp").to_str().unwrap()])
        .args(["--connect", &db_port.to_string()])
        .env("DATABASE_URL", format!("postgres://u:p@127.0.0.1:{db_port}/db"))
        .env("SANDBOX", mode)
        .env("BIND_ADDR", "127.0.0.1:18081")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let lines = stdout.lines().map(String::from).collect();
    (out.status.success(), lines, String::from_utf8_lossy(&out.stderr).to_string() + &stdout)
}

fn verdict(lines: &[String], needle: &str) -> String {
    lines
        .iter()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("sem resultado para {needle}: {lines:?}"))
        .rsplit(' ')
        .next()
        .unwrap()
        .to_string()
}

fn skip_if_unsupported(ok: bool, log: &str) -> bool {
    if !ok && (log.contains("sandbox exigido") || log.contains("seccomp")) {
        eprintln!("kernel sem Landlock/seccomp completos: teste ignorado\n{log}");
        return true;
    }
    false
}

#[test]
fn worker_profile_is_locked_down() {
    let s = setup();
    let (ok, lines, log) = run("worker", "required", &s);
    if skip_if_unsupported(ok, &log) {
        return;
    }
    assert!(ok, "{log}");

    // Arquivos: nada fora do próprio job e do sistema.
    assert_eq!(verdict(&lines, s.secret.path().to_str().unwrap()), "DENIED", "leu segredo");
    assert_eq!(verdict(&lines, "contrato.pdf"), "DENIED", "leu o upload de outro job");
    assert_eq!(verdict(&lines, "plantado.tmp"), "DENIED", "gravou no diretório de outro job");
    assert_eq!(verdict(&lines, "/environ"), "DENIED", "leu o ambiente do processo pai");
    assert_eq!(verdict(&lines, "saida.tmp"), "OK", "precisa gravar no próprio job");
    assert_eq!(verdict(&lines, "/etc/os-release"), "OK");

    // Rede: nem a porta do banco (que a API usa) é alcançável.
    assert_eq!(verdict(&lines, "connect "), "DENIED", "worker conectou no banco");

    // Chamadas de sistema.
    assert_eq!(verdict(&lines, "syscall execve"), "DENIED", "worker executou programa");
    assert_eq!(verdict(&lines, "syscall socket-inet"), "DENIED", "worker abriu socket de rede");
    assert_eq!(verdict(&lines, "syscall process_vm_readv"), "DENIED");
    assert_eq!(verdict(&lines, "syscall unshare"), "DENIED");

    // Limites de recursos. Memória: o worker é a primeira vítima do OOM killer.
    assert_eq!(verdict(&lines, "oom_score_adj"), "1000");
    assert_ne!(verdict(&lines, "rlimit cpu"), "unlimited", "sem teto de CPU");
    assert_eq!(verdict(&lines, "rlimit core"), "0");
    assert_eq!(verdict(&lines, "rlimit nofile"), "1024");
    // Sem core dump nem /proc legível por outros processos (o core teria o documento).
    assert_eq!(verdict(&lines, "dumpable"), "false");
}

#[test]
fn without_sandbox_the_same_operations_work() {
    // Controle: prova que as negações acima vêm do confinamento, e não do ambiente do teste.
    let s = setup();
    let (ok, lines, log) = run("worker", "off", &s);
    assert!(ok, "{log}");
    assert_eq!(verdict(&lines, s.secret.path().to_str().unwrap()), "OK");
    assert_eq!(verdict(&lines, "contrato.pdf"), "OK");
    assert_eq!(verdict(&lines, "syscall execve"), "OK");
    assert_eq!(verdict(&lines, "syscall socket-inet"), "OK");
    assert_eq!(verdict(&lines, "syscall process_vm_readv"), "OK");
}

#[test]
fn api_profile_keeps_what_the_api_needs_and_blocks_the_rest() {
    let s = setup();
    let (ok, lines, log) = run("api", "required", &s);
    if skip_if_unsupported(ok, &log) {
        return;
    }
    assert!(ok, "{log}");
    // Precisa: executar o worker, abrir sockets, falar com o banco.
    assert_eq!(verdict(&lines, "syscall execve"), "OK");
    assert_eq!(verdict(&lines, "syscall socket-inet"), "OK");
    assert!(verdict(&lines, "connect ").starts_with("OK"), "porta do banco deve estar liberada");
    // Não precisa: espionar processos, criar namespaces.
    assert_eq!(verdict(&lines, "syscall process_vm_readv"), "DENIED");
    assert_eq!(verdict(&lines, "syscall unshare"), "DENIED");
    assert_eq!(verdict(&lines, s.secret.path().to_str().unwrap()), "DENIED");
    // Memória e ambiente da API inacessíveis a outros processos do mesmo usuário.
    assert_eq!(verdict(&lines, "dumpable"), "false");
}

#[test]
fn worker_confines_itself_inside_the_api_sandbox() {
    // Produção: a API (já sob Landlock + seccomp) executa o worker, que aplica o próprio
    // confinamento POR CIMA. Aqui os dois sandboxes são aninhados de verdade.
    let s = setup();
    let db_port = s.db_listener.local_addr().unwrap().port();
    let out = Command::new(env!("CARGO_BIN_EXE_api_ocr"))
        .arg("sandbox-check")
        .args(["--profile", "api", "--nested-worker"])
        .args(["--job-dir", s.job.path().to_str().unwrap()])
        .args(["--read", s.other_job.path().join("contrato.pdf").to_str().unwrap()])
        .args(["--write", s.job.path().join("saida.tmp").to_str().unwrap()])
        .args(["--connect", &db_port.to_string()])
        .env("DATABASE_URL", format!("postgres://u:p@127.0.0.1:{db_port}/db"))
        .env("SANDBOX", "required")
        .env("BIND_ADDR", "127.0.0.1:18082")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let log = String::from_utf8_lossy(&out.stderr).to_string() + &stdout;
    if skip_if_unsupported(out.status.success(), &log) {
        return;
    }
    assert!(out.status.success(), "{log}");
    let nested: Vec<String> = stdout.lines().filter_map(|l| l.strip_prefix("nested ").map(String::from)).collect();
    assert!(!nested.iter().any(|l| l.starts_with("FAILED")), "worker não conseguiu se confinar dentro da API: {log}");

    // A API (fora) vê o banco e o diretório temporário; o worker (dentro), não.
    let api: Vec<String> = stdout.lines().filter(|l| !l.starts_with("nested ")).map(String::from).collect();
    assert!(verdict(&api, "connect ").starts_with("OK"));
    assert_eq!(verdict(&nested, "connect "), "DENIED");
    assert_eq!(verdict(&nested, "contrato.pdf"), "DENIED");
    assert_eq!(verdict(&nested, "saida.tmp"), "OK");
    assert_eq!(verdict(&nested, "syscall execve"), "DENIED");
    assert_eq!(verdict(&nested, "oom_score_adj"), "1000", "o worker precisa ser a 1ª vítima do OOM killer");
    assert_eq!(verdict(&nested, "dumpable"), "false");
}
