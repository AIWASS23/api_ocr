mod common;
use std::time::Duration;

use common::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Abre uma requisição que fica "pendurada" lendo o corpo (Content-Length maior que o enviado).
async fn stalled_request(addr: std::net::SocketAddr) -> TcpStream {
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(
        b"POST /oauth/token HTTP/1.1\r\nHost: x\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: 1000\r\n\r\ngrant_",
    )
    .await
    .unwrap();
    s
}

#[tokio::test]
async fn load_shedding_returns_503_when_in_flight_cap_is_reached_and_recovers() {
    let app = TestApp::start_with(|c| c.max_in_flight = 2).await;
    let a = stalled_request(app.addr).await;
    let b = stalled_request(app.addr).await;
    tokio::time::sleep(Duration::from_millis(300)).await;

    let r = app.http.get(app.url("/healthz")).send().await.unwrap();
    assert_eq!(r.status(), 503, "com o teto de requisições em andamento atingido, deve recusar");
    assert!(r.headers().contains_key("retry-after"));
    assert_eq!(r.headers()["x-content-type-options"], "nosniff", "a recusa também leva os cabeçalhos de segurança");

    drop((a, b));
    tokio::time::sleep(Duration::from_millis(500)).await;
    let r = app.http.get(app.url("/healthz")).send().await.unwrap();
    assert_eq!(r.status(), 200, "libera quando as requisições pendentes terminam");
}

#[tokio::test]
async fn stalled_body_is_cut_by_idle_timeout() {
    let app = TestApp::start().await;
    let mut s = stalled_request(app.addr).await;
    let mut buf = [0u8; 512];
    let n = tokio::time::timeout(Duration::from_secs(25), s.read(&mut buf))
        .await
        .expect("o servidor deveria encerrar o corpo ocioso")
        .unwrap_or(0);
    let text = String::from_utf8_lossy(&buf[..n]);
    assert!(n == 0 || text.contains("408") || text.contains("400") || text.contains("500"), "{text}");
}

async fn h2_connect(addr: std::net::SocketAddr) -> (h2::client::SendRequest<bytes::Bytes>, tokio::task::JoinHandle<Result<(), h2::Error>>) {
    let tcp = TcpStream::connect(addr).await.unwrap();
    let (client, conn) = h2::client::handshake(tcp).await.unwrap();
    (client, tokio::spawn(conn))
}

#[tokio::test]
async fn http2_advertises_stream_limits_and_serves_requests() {
    let app = TestApp::start().await;
    let (client, _conn) = h2_connect(app.addr).await;
    let mut client = client.ready().await.unwrap();

    let (resp, _) = client.send_request(http::Request::get("http://x/healthz").body(()).unwrap(), true).unwrap();
    let resp = resp.await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["x-frame-options"], "DENY");

    assert_eq!(client.current_max_send_streams(), 32, "SETTINGS_MAX_CONCURRENT_STREAMS anunciado pelo servidor");
}

#[tokio::test]
async fn http2_rapid_reset_flood_is_cut_off_and_server_stays_up() {
    let app = TestApp::start().await;
    let (client, conn) = h2_connect(app.addr).await;
    let mut client = client.ready().await.unwrap();

    // Abre e cancela streams o mais rápido possível (padrão do "rapid reset").
    let mut sent = 0usize;
    for _ in 0..5000 {
        let req = http::Request::get("http://x/healthz").body(()).unwrap();
        match client.send_request(req, false) {
            Ok((_resp, mut stream)) => {
                stream.send_reset(h2::Reason::CANCEL);
                sent += 1;
            }
            Err(_) => break, // servidor já fechou a conexão
        }
        if sent.is_multiple_of(64) {
            tokio::task::yield_now().await;
        }
    }

    // O servidor deve ter encerrado a conexão abusiva (GOAWAY) em poucos segundos.
    let ended = tokio::time::timeout(Duration::from_secs(10), conn).await;
    assert!(ended.is_ok(), "a conexão abusiva não foi encerrada pelo servidor (enviou {sent} streams)");

    // E o serviço continua respondendo a clientes normais.
    for _ in 0..3 {
        assert_eq!(app.http.get(app.url("/healthz")).send().await.unwrap().status(), 200);
    }
}
