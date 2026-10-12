mod common;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use common::*;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

fn redis_url() -> String {
    std::env::var("TEST_REDIS_URL").expect("defina TEST_REDIS_URL (ver api/README.md)")
}

async fn token_calls(app: &TestApp, n: usize) -> Vec<u16> {
    let mut out = vec![];
    for _ in 0..n {
        let r = app
            .http
            .post(app.url("/oauth/token"))
            .form(&[("grant_type", "refresh_token"), ("client_id", CLIENT_ID), ("refresh_token", "x")])
            .send()
            .await
            .unwrap();
        out.push(r.status().as_u16());
    }
    out
}

#[tokio::test]
async fn limit_is_global_across_replicas() {
    let url = redis_url();
    let prefix = format!("t{}:", Uuid::new_v4().simple());
    let tweak = |url: String, prefix: String| {
        move |c: &mut api_ocr::config::Config| {
            c.rate_auth_per_min = 10; // rajada de 5, em CADA réplica
            c.rate_limit_redis_url = Some(url);
            c.rate_limit_redis_prefix = prefix;
        }
    };
    let a = TestApp::start_with(tweak(url.clone(), prefix.clone())).await;
    let b = TestApp::start_with(tweak(url, prefix)).await;

    // 3 chamadas em cada réplica: com limites só locais (5 por réplica) todas passariam.
    let from_a = token_calls(&a, 3).await;
    let from_b = token_calls(&b, 3).await;
    let all: Vec<u16> = from_a.iter().chain(&from_b).copied().collect();
    let limited = all.iter().filter(|&&s| s == 429).count();
    assert!(limited >= 1, "o limite global (5) deveria barrar ao menos 1 das 6 chamadas: {all:?}");
    assert_eq!(all.iter().filter(|&&s| s != 429).count(), 5, "exatamente a rajada global passa: {all:?}");

    // Controle: sem Redis, as mesmas 6 chamadas passam (cada réplica tem sua rajada de 5).
    let c = TestApp::start_with(|c| c.rate_auth_per_min = 10).await;
    let d = TestApp::start_with(|c| c.rate_auth_per_min = 10).await;
    let mut local: Vec<u16> = token_calls(&c, 3).await;
    local.extend(token_calls(&d, 3).await);
    assert!(local.iter().all(|&s| s != 429), "{local:?}");
}

#[tokio::test]
async fn global_limit_is_per_key_not_shared_between_prefixes() {
    let url = redis_url();
    let mk = |prefix: String| {
        let url = url.clone();
        move |c: &mut api_ocr::config::Config| {
            c.rate_auth_per_min = 10;
            c.rate_limit_redis_url = Some(url);
            c.rate_limit_redis_prefix = prefix;
        }
    };
    let a = TestApp::start_with(mk(format!("t{}:", Uuid::new_v4().simple()))).await;
    let b = TestApp::start_with(mk(format!("t{}:", Uuid::new_v4().simple()))).await;
    assert!(token_calls(&a, 5).await.iter().all(|&s| s != 429));
    assert!(token_calls(&b, 5).await.iter().all(|&s| s != 429), "prefixos diferentes não compartilham contadores");
}

#[tokio::test]
async fn unreachable_redis_at_startup_is_a_hard_error() {
    let mut cfg = base_config("postgres://nao-usado".into());
    cfg.rate_limit_redis_url = Some("redis://:x@127.0.0.1:1/".into());
    let admin = admin_url();
    let db = sqlx::PgPool::connect_lazy(&admin).unwrap();
    let err = api_ocr::build_state(cfg, db).await.err().expect("deveria falhar");
    assert!(err.to_string().contains("RATE_LIMIT_REDIS_URL"), "{err}");
}

/// Proxy TCP que pode ser derrubado (inclusive as conexões já abertas).
async fn start_proxy(target: SocketAddr) -> (SocketAddr, CancellationToken) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let stop = CancellationToken::new();
    let st = stop.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                () = st.cancelled() => break,
                Ok((mut inbound, _)) = listener.accept() => {
                    let st = st.clone();
                    tokio::spawn(async move {
                        let Ok(mut out) = TcpStream::connect(target).await else { return };
                        tokio::select! {
                            () = st.cancelled() => {},
                            _ = tokio::io::copy_bidirectional(&mut inbound, &mut out) => {},
                        }
                    });
                }
            }
        }
    });
    (addr, stop)
}

#[tokio::test]
async fn redis_outage_fails_open_to_local_limits_without_slowing_requests() {
    let real = url::Url::parse(&redis_url()).unwrap();
    let target: SocketAddr = format!("{}:{}", real.host_str().unwrap(), real.port().unwrap()).parse().unwrap();
    let (proxy_addr, stop) = start_proxy(target).await;
    let mut via_proxy = real.clone();
    via_proxy.set_host(Some("127.0.0.1")).unwrap();
    via_proxy.set_port(Some(proxy_addr.port())).unwrap();

    let prefix = format!("t{}:", Uuid::new_v4().simple());
    let app = TestApp::start_with(move |c| {
        c.rate_auth_per_min = 20; // rajada local de 10
        c.rate_limit_redis_url = Some(via_proxy.to_string());
        c.rate_limit_redis_prefix = prefix;
    })
    .await;

    // Redis no ar: funciona.
    assert!(token_calls(&app, 2).await.iter().all(|&s| s != 429));

    // Derruba o Redis. A API segue respondendo, sem ficar lenta, e continua limitando localmente.
    stop.cancel();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let started = Instant::now();
    let during = token_calls(&app, 12).await;
    let elapsed = started.elapsed();
    assert!(during.iter().all(|&s| s != 0), "sem erro de conexão");
    assert!(during.contains(&429), "o limite LOCAL continua valendo sem Redis: {during:?}");
    assert!(during.iter().any(|&s| s != 429), "{during:?}");
    assert!(elapsed < Duration::from_secs(4), "o disjuntor deve evitar esperar o Redis a cada requisição ({elapsed:?})");
    assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
}
