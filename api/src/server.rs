//! Laço de aceitação de conexões com defesas contra exaustão de recursos:
//! teto global de conexões, teto por IP, timeout de leitura de cabeçalhos
//! (slowloris), limites de HTTP/2 e encerramento gracioso.

use std::collections::HashMap;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::ConnectInfo;
use hyper::body::Incoming;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder;
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tower::ServiceExt;

use crate::config::Config;

const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);
const GRACE_PERIOD: Duration = Duration::from_secs(30);

struct IpSlot {
    map: Arc<Mutex<HashMap<IpAddr, usize>>>,
    ip: IpAddr,
}

impl IpSlot {
    fn acquire(map: &Arc<Mutex<HashMap<IpAddr, usize>>>, ip: IpAddr, max: usize) -> Option<Self> {
        let mut m = map.lock().unwrap_or_else(PoisonError::into_inner);
        let n = m.entry(ip).or_insert(0);
        if *n >= max {
            return None;
        }
        *n += 1;
        Some(Self { map: map.clone(), ip })
    }
}

impl Drop for IpSlot {
    fn drop(&mut self) {
        let mut m = self.map.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(n) = m.get_mut(&self.ip) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                m.remove(&self.ip);
            }
        }
    }
}

fn connection_builder() -> Builder<TokioExecutor> {
    let mut b = Builder::new(TokioExecutor::new());
    b.http1()
        .timer(TokioTimer::new())
        .header_read_timeout(HEADER_READ_TIMEOUT)
        .keep_alive(true)
        .max_buf_size(32 * 1024);
    b.http2()
        .timer(TokioTimer::new())
        .max_concurrent_streams(32)
        .keep_alive_interval(Some(Duration::from_secs(30)))
        .keep_alive_timeout(Duration::from_secs(10))
        .max_header_list_size(32 * 1024)
        // Defesa contra "rapid reset" (CVE-2023-44487) e afins: explícito, não depende do padrão.
        .max_pending_accept_reset_streams(20)
        .max_local_error_reset_streams(128)
        .initial_stream_window_size(64 * 1024)
        .initial_connection_window_size(256 * 1024)
        .max_frame_size(16 * 1024)
        .max_send_buf_size(256 * 1024);
    b
}

pub async fn serve(listener: TcpListener, app: Router, cfg: &Config, shutdown: impl Future<Output = ()>) {
    let conn_limit = Arc::new(Semaphore::new(cfg.max_connections));
    let per_ip: Arc<Mutex<HashMap<IpAddr, usize>>> = Arc::default();
    let graceful = GracefulShutdown::new();
    let builder = connection_builder();
    // Atrás de proxy todas as conexões vêm do mesmo IP: o teto por IP fica com o proxy.
    let per_ip_limit = (!cfg.trust_proxy).then_some(cfg.max_connections_per_ip);
    tokio::pin!(shutdown);

    loop {
        let (stream, peer) = tokio::select! {
            r = listener.accept() => match r {
                Ok(x) => x,
                Err(e) => {
                    tracing::warn!(error = %e, "falha em accept()");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            },
            () = &mut shutdown => break,
        };

        // Sem vaga: fecha na hora (melhor que enfileirar sob ataque).
        let Ok(permit) = conn_limit.clone().try_acquire_owned() else {
            drop(stream);
            continue;
        };
        let slot = match per_ip_limit {
            Some(max) => match IpSlot::acquire(&per_ip, peer.ip(), max) {
                Some(s) => Some(s),
                None => {
                    drop(stream);
                    continue;
                }
            },
            None => None,
        };
        let _ = stream.set_nodelay(true);

        let app = app.clone();
        let service = TowerToHyperService::new(tower::service_fn(move |mut req: hyper::Request<Incoming>| {
            req.extensions_mut().insert(ConnectInfo::<SocketAddr>(peer));
            app.clone().oneshot(req.map(Body::new))
        }));

        let conn = builder.serve_connection_with_upgrades(TokioIo::new(stream), service);
        let conn = graceful.watch(conn.into_owned());
        tokio::spawn(async move {
            let _permit = permit;
            let _slot = slot;
            if let Err(e) = conn.await {
                tracing::debug!(error = %e, %peer, "conexão encerrada com erro");
            }
        });
    }

    tracing::info!("encerrando: aguardando conexões ativas");
    tokio::select! {
        () = graceful.shutdown() => tracing::info!("todas as conexões encerradas"),
        () = tokio::time::sleep(GRACE_PERIOD) => tracing::warn!("prazo de encerramento esgotado"),
    }
}
