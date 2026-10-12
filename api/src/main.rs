use std::io::BufRead;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use api_ocr::hardening::{self, Profile};
use api_ocr::sandbox::{self, Mode, Outcome};
use api_ocr::{config::Config, crypto, server, store, validate, worker};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "api_ocr", about = "API de OCR com servidor OAuth2 próprio")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Sobe o servidor HTTP (aplica as migrações antes, salvo --no-migrate).
    Serve {
        #[arg(long)]
        no_migrate: bool,
    },
    /// Aplica as migrações do banco.
    Migrate,
    /// Gera um segredo aleatório (para JWT_SECRET / PASSWORD_PEPPER).
    GenSecret,
    /// Cria um usuário. A senha é lida da primeira linha do stdin.
    CreateUser {
        #[arg(long)]
        email: String,
    },
    /// Sonda `GET /healthz` em 127.0.0.1 (para HEALTHCHECK de contêiner sem curl). Sai 0 se saudável.
    Healthcheck,
    /// Aplica o sandbox e tenta operações proibidas/permitidas (diagnóstico e testes).
    SandboxCheck {
        /// Perfil a aplicar: o da API ou o do worker de OCR.
        #[arg(long, value_enum, default_value = "api")]
        profile: ProbeProfile,
        /// (perfil worker) diretório do job, o único gravável.
        #[arg(long)]
        job_dir: Option<std::path::PathBuf>,
        #[arg(long = "read")]
        read: Vec<std::path::PathBuf>,
        #[arg(long = "write")]
        write: Vec<std::path::PathBuf>,
        #[arg(long = "connect")]
        connect: Vec<u16>,
        /// (perfil api) roda também a sonda do worker COMO PROCESSO FILHO, aninhando os dois
        /// sandboxes exatamente como em produção; as linhas saem com o prefixo `nested `.
        #[arg(long)]
        nested_worker: bool,
    },
    /// (interno) Processa um documento num processo isolado. Executado pela própria API.
    #[command(hide = true)]
    OcrWorker {
        #[arg(long)]
        file: Option<std::path::PathBuf>,
        /// Só pré-carrega os modelos (e grava o cache de grafos).
        #[arg(long)]
        warmup: bool,
        #[arg(long, default_value_t = 100)]
        max_pages: usize,
        #[arg(long, default_value_t = 180)]
        timeout_secs: u64,
        #[arg(long, default_value_t = 0)]
        memory_mb: u64,
    },
    /// Ativa o segundo fator (TOTP) de um usuário e mostra a URI para o app autenticador.
    MfaEnable {
        #[arg(long)]
        email: String,
    },
    /// Desativa o segundo fator de um usuário.
    MfaDisable {
        #[arg(long)]
        email: String,
    },
    /// Derruba na hora todas as sessões de um usuário (access e refresh tokens).
    RevokeSessions {
        #[arg(long)]
        email: String,
    },
    /// Remove os bloqueios de login de um usuário.
    UnlockUser {
        #[arg(long)]
        email: String,
    },
    /// Registra um cliente OAuth2 público (PKCE).
    CreateClient {
        #[arg(long)]
        id: String,
        #[arg(long)]
        name: String,
        /// Repita para várias URIs (https, ou http em loopback).
        #[arg(long = "redirect-uri", required = true)]
        redirect_uris: Vec<String>,
        #[arg(long = "scope", default_values_t = ["ocr".to_string(), "profile".to_string()])]
        scopes: Vec<String>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ProbeProfile {
    Api,
    Worker,
}

/// Segredos que só a configuração precisa. Saem do ambiente do processo logo depois de lidos:
/// nada que a API execute depois (o worker já recebe um ambiente zerado) os herda por engano.
/// (O `/proc/<pid>/environ` original fica protegido pelo `PR_SET_DUMPABLE=0`.)
const SECRET_VARS: &[&str] = &["JWT_SECRET", "JWT_SECRET_PREVIOUS", "PASSWORD_PEPPER", "DATABASE_URL", "RATE_LIMIT_REDIS_URL"];

fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,sqlx=warn"));
    let builder = tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr);
    if std::env::var("LOG_FORMAT").is_ok_and(|v| v.eq_ignore_ascii_case("json")) {
        builder.json().flatten_event(true).with_current_span(true).init();
    } else {
        builder.init();
    }
}

/// Landlock + seccomp + não-depurável no processo da API. Precisa vir ANTES do runtime: os dois
/// sandboxes só alcançam threads criadas depois.
fn harden_api(plan: &sandbox::Plan) -> anyhow::Result<()> {
    let mode = Mode::from_env().map_err(|e| anyhow::anyhow!(e))?;
    match sandbox::apply(plan, mode)? {
        Outcome::Enforced => tracing::info!("sandbox Landlock aplicado por completo"),
        Outcome::Partial => tracing::warn!("sandbox Landlock aplicado PARCIALMENTE (kernel antigo)"),
        Outcome::Unsupported => tracing::warn!("kernel sem Landlock: a API roda SEM sandbox de arquivos/rede"),
        Outcome::Disabled => tracing::warn!("SANDBOX=off: a API roda SEM sandbox de processo"),
    }
    if mode != Mode::Off {
        match hardening::apply_seccomp(Profile::Api) {
            Ok(()) => tracing::info!("filtro seccomp aplicado"),
            Err(e) if mode == Mode::Required => return Err(e.into()),
            Err(e) => tracing::warn!(error = %e, "seccomp indisponível"),
        }
    }
    hardening::set_not_dumpable()?;
    Ok(())
}

fn main() -> anyhow::Result<ExitCode> {
    init_logging();
    let cli = Cli::parse();

    match cli.cmd {
        // O worker se confina sozinho e não usa runtime assíncrono.
        Cmd::OcrWorker { file, warmup, max_pages, timeout_secs, memory_mb } => {
            return Ok(worker::main(worker::WorkerArgs {
                file,
                warmup,
                max_pages,
                timeout: std::time::Duration::from_secs(timeout_secs),
                memory_mb,
            }));
        }
        Cmd::SandboxCheck { profile, job_dir, read, write, connect, nested_worker } => {
            match profile {
                ProbeProfile::Api => harden_api(&sandbox::plan_from_env())?,
                ProbeProfile::Worker => {
                    let args = worker::WorkerArgs { file: None, warmup: false, max_pages: 1, timeout: std::time::Duration::from_secs(60), memory_mb: 0 };
                    worker::confine(job_dir.as_deref(), sandbox::trusted_graph_dir().as_deref(), false, &worker::limits_for(&args))?;
                }
            }
            for line in sandbox::probe(&read, &write, &connect).into_iter().chain(hardening::probe_syscalls()) {
                println!("{line}");
            }
            if nested_worker {
                let mut child = std::process::Command::new(std::env::current_exe()?);
                child.args(["sandbox-check", "--profile", "worker"]);
                if let Some(dir) = &job_dir {
                    child.arg("--job-dir").arg(dir);
                }
                for p in &read {
                    child.arg("--read").arg(p);
                }
                for p in &write {
                    child.arg("--write").arg(p);
                }
                for p in &connect {
                    child.args(["--connect", &p.to_string()]);
                }
                let out = child.output()?;
                for line in String::from_utf8_lossy(&out.stdout).lines() {
                    println!("nested {line}");
                }
                if !out.status.success() {
                    println!("nested FAILED {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or(""));
                }
            }
            return Ok(ExitCode::SUCCESS);
        }
        Cmd::Serve { no_migrate } => {
            let cfg = Config::from_env()?;
            let plan = sandbox::plan_from_env(); // lê DATABASE_URL etc.: antes de limpar o ambiente
            for k in SECRET_VARS {
                // SAFETY: ainda não há outras threads (o runtime é criado abaixo).
                unsafe { std::env::remove_var(k) };
            }
            harden_api(&plan)?;
            tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(serve(cfg, no_migrate))?;
        }
        cmd => tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(run(cmd))?,
    }
    Ok(ExitCode::SUCCESS)
}

async fn serve(cfg: Config, no_migrate: bool) -> anyhow::Result<()> {
    tracing::info!(?cfg, "configuração carregada");
    // O banco pode ainda estar subindo (Kubernetes/compose): espera até 2 min em vez de cair e
    // ser reiniciado em laço.
    let mut attempt = 0u32;
    let db = loop {
        match api_ocr::connect_db(&cfg.database_url).await {
            Ok(db) => break db,
            Err(e) if attempt < 40 => {
                attempt += 1;
                tracing::warn!(error = %e, attempt, "banco indisponível; tentando de novo em 3 s");
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
            Err(e) => return Err(e),
        }
    };
    if !no_migrate {
        api_ocr::MIGRATOR.run(&db).await?;
    }
    if cfg.enable_docs {
        api_ocr::docs::ensure_client(&db, &cfg.issuer).await?;
        tracing::info!("documentação ligada: {}/docs", cfg.issuer);
    }
    let addr = cfg.bind_addr;
    worker::spawn_warmup(cfg.clone());
    let st = api_ocr::build_state(cfg.clone(), db).await?;
    api_ocr::spawn_maintenance(st.clone());
    let app = api_ocr::router(st);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "servidor no ar");
    server::serve(listener, app, &cfg, shutdown_signal()).await;
    Ok(())
}

/// Usuário pelo e-mail (para os comandos do operador).
async fn user_by_email(db: &sqlx::PgPool, email: &str) -> anyhow::Result<uuid::Uuid> {
    let email = validate::normalize_email(email).ok_or_else(|| anyhow::anyhow!("e-mail inválido"))?;
    store::find_user_id_by_email(db, &email).await?.ok_or_else(|| anyhow::anyhow!("usuário não encontrado"))
}

async fn run(cmd: Cmd) -> anyhow::Result<()> {
    match cmd {
        Cmd::Serve { .. } | Cmd::SandboxCheck { .. } | Cmd::OcrWorker { .. } => unreachable!("tratados em main"),
        Cmd::MfaEnable { email } => {
            let cfg = Config::from_env()?;
            let db = api_ocr::connect_db(&cfg.database_url).await?;
            let id = user_by_email(&db, &email).await?;
            let secret = api_ocr::mfa::new_secret();
            let blob = api_ocr::mfa::MfaCipher::from_pepper(&cfg.password_pepper).encrypt(id, &secret);
            store::set_totp_secret(&db, id, Some(&blob)).await?;
            let issuer = url::Url::parse(&cfg.issuer).ok().and_then(|u| u.host_str().map(str::to_string)).unwrap_or_else(|| "API OCR".into());
            println!("MFA ativado. Cadastre no app autenticador (gere o QR code a partir da URI):");
            println!("{}", api_ocr::mfa::otpauth_uri(&issuer, &email, &secret));
            println!("chave manual: {}", api_ocr::mfa::base32(&secret));
            Ok(())
        }
        Cmd::MfaDisable { email } => {
            let db = api_ocr::connect_db(&std::env::var("DATABASE_URL")?).await?;
            let id = user_by_email(&db, &email).await?;
            store::set_totp_secret(&db, id, None).await?;
            println!("MFA desativado");
            Ok(())
        }
        Cmd::RevokeSessions { email } => {
            let db = api_ocr::connect_db(&std::env::var("DATABASE_URL")?).await?;
            let id = user_by_email(&db, &email).await?;
            store::revoke_all_sessions(&db, id).await?;
            store::audit(&db, "sessions_revoked", Some(id), None, None, Some("comando do operador")).await;
            println!("sessões derrubadas");
            Ok(())
        }
        Cmd::UnlockUser { email } => {
            let db = api_ocr::connect_db(&std::env::var("DATABASE_URL")?).await?;
            let id = user_by_email(&db, &email).await?;
            store::unlock_user(&db, id).await?;
            println!("bloqueios removidos");
            Ok(())
        }
        Cmd::Healthcheck => {
            use std::io::{Read, Write};
            let port: u16 = std::env::var("BIND_ADDR").ok().and_then(|a| a.rsplit(':').next().and_then(|p| p.parse().ok())).unwrap_or(8080);
            let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
            let mut s = std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(2))?;
            s.set_read_timeout(Some(std::time::Duration::from_secs(2)))?;
            s.write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
            let mut buf = String::new();
            s.read_to_string(&mut buf)?;
            anyhow::ensure!(buf.starts_with("HTTP/1.1 200"), "não saudável: {}", buf.lines().next().unwrap_or(""));
            Ok(())
        }
        Cmd::GenSecret => {
            println!("{}", crypto::random_token(48));
            Ok(())
        }
        Cmd::Migrate => {
            let db = api_ocr::connect_db(&std::env::var("DATABASE_URL")?).await?;
            api_ocr::MIGRATOR.run(&db).await?;
            println!("migrações aplicadas");
            Ok(())
        }
        Cmd::CreateUser { email } => {
            let cfg = Config::from_env()?;
            let email = validate::normalize_email(&email).ok_or_else(|| anyhow::anyhow!("e-mail inválido"))?;
            let mut password = String::new();
            std::io::stdin().lock().read_line(&mut password)?;
            let password = password.trim_end_matches(['\r', '\n']).to_string();
            validate::check_password(&password, &email).map_err(|m| anyhow::anyhow!(m))?;
            let db = api_ocr::connect_db(&cfg.database_url).await?;
            let svc = crypto::PasswordService::new(cfg.password_pepper.clone(), 1).map_err(|e| anyhow::anyhow!("{e}"))?;
            let hash = svc.hash(password).await.map_err(|e| anyhow::anyhow!("{e}"))?;
            let id = uuid::Uuid::new_v4();
            if !store::create_user(&db, id, &email, &hash).await? {
                anyhow::bail!("já existe um usuário com esse e-mail");
            }
            println!("usuário criado: {id}");
            Ok(())
        }
        Cmd::CreateClient { id, name, redirect_uris, scopes } => {
            let url = std::env::var("DATABASE_URL")?;
            anyhow::ensure!(validate::valid_client_id(&id), "client id inválido (3-64 de A-Za-z0-9._-)");
            anyhow::ensure!(!name.trim().is_empty() && name.len() <= 100, "nome inválido");
            for u in &redirect_uris {
                anyhow::ensure!(validate::valid_redirect_uri(u), "redirect_uri inválida: {u}");
            }
            for s in &scopes {
                anyhow::ensure!(validate::SUPPORTED_SCOPES.contains(&s.as_str()), "escopo desconhecido: {s}");
            }
            let db = api_ocr::connect_db(&url).await?;
            if !store::create_client(&db, &id, name.trim(), &redirect_uris, &scopes).await? {
                anyhow::bail!("já existe um cliente com esse id");
            }
            println!("cliente criado: {id}");
            Ok(())
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! { () = ctrl_c => {}, () = term => {} }
}
