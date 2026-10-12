use std::io::Write;
use std::net::TcpListener;
use std::process::Command;

struct Probe {
    secret: tempfile::NamedTempFile,
    secret_dir_file: std::path::PathBuf,
    forbidden_port: u16,
    _listener: TcpListener,
    db_listener: TcpListener,
}

fn setup() -> Probe {
    // "Segredo" FORA das áreas permitidas (o diretório do pacote não está no plano).
    let mut secret = tempfile::NamedTempFile::new_in(env!("CARGO_MANIFEST_DIR")).unwrap();
    writeln!(secret, "JWT_SECRET=super-secreto").unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let db_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    Probe {
        secret_dir_file: std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("escrita-proibida-{}.tmp", std::process::id())),
        forbidden_port: listener.local_addr().unwrap().port(),
        _listener: listener,
        db_listener,
        secret,
    }
}

fn run(mode: &str, p: &Probe) -> (bool, Vec<String>, String) {
    let home = std::env::var("HOME").unwrap();
    let tmp_ok = std::env::temp_dir().join(format!("sandbox-ok-{}.tmp", std::process::id()));
    let out = Command::new(env!("CARGO_BIN_EXE_api_ocr"))
        .arg("sandbox-check")
        .args(["--read", p.secret.path().to_str().unwrap()])
        .args(["--read", &format!("{home}/.bashrc")])
        .args(["--read", "/etc/os-release"])
        .args(["--write", p.secret_dir_file.to_str().unwrap()])
        .args(["--write", tmp_ok.to_str().unwrap()])
        .args(["--connect", &p.forbidden_port.to_string()])
        .args(["--connect", &p.db_listener.local_addr().unwrap().port().to_string()])
        // O "banco" é a porta do db_listener: a única conexão permitida.
        .env("DATABASE_URL", format!("postgres://u:p@127.0.0.1:{}/db", p.db_listener.local_addr().unwrap().port()))
        .env("SANDBOX", mode)
        .env("BIND_ADDR", "127.0.0.1:18080")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let lines = stdout
        .lines()
        .filter(|l| l.starts_with("read ") || l.starts_with("write ") || l.starts_with("connect "))
        .map(String::from)
        .collect();
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

#[test]
fn without_sandbox_the_forbidden_operations_work() {
    let p = setup();
    let (ok, lines, log) = run("off", &p);
    assert!(ok, "{log}");
    // Controle: sem o Landlock, tudo "proibido" funciona. Prova que a negação abaixo vem do sandbox.
    assert_eq!(verdict(&lines, p.secret.path().to_str().unwrap()), "OK");
    assert_eq!(verdict(&lines, "escrita-proibida"), "OK");
    assert!(verdict(&lines, &format!("connect {}", p.forbidden_port)).starts_with("OK"));
}

#[test]
fn with_sandbox_secrets_writes_and_foreign_ports_are_blocked_but_the_app_still_works() {
    let p = setup();
    let (ok, lines, log) = run("required", &p);
    if !ok && log.contains("sandbox exigido") {
        eprintln!("kernel sem Landlock completo: teste ignorado");
        return;
    }
    assert!(ok, "{log}");

    // BLOQUEADO
    assert_eq!(verdict(&lines, p.secret.path().to_str().unwrap()), "DENIED", "leu um segredo fora da área permitida");
    assert_eq!(verdict(&lines, ".bashrc"), "DENIED", "leu arquivo do HOME");
    assert_eq!(verdict(&lines, "escrita-proibida"), "DENIED", "gravou fora do diretório temporário");
    assert_eq!(verdict(&lines, &format!("connect {}", p.forbidden_port)), "DENIED", "conectou em porta não permitida");

    // PERMITIDO (o app precisa continuar funcionando)
    assert_eq!(verdict(&lines, "/etc/os-release"), "OK");
    assert_eq!(verdict(&lines, "sandbox-ok"), "OK", "precisa gravar no diretório temporário");
    assert!(verdict(&lines, &format!("connect {}", p.db_listener.local_addr().unwrap().port())).starts_with("OK"), "porta do banco deve estar liberada");
}
