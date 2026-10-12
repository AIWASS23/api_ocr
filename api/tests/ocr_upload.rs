mod common;
use std::io::Write;
use std::time::{Duration, Instant};

use common::*;
use reqwest::multipart::{Form, Part};
use serde_json::Value;

async fn upload(app: &TestApp, token: &str, name: &str, bytes: Vec<u8>) -> reqwest::Response {
    let form = Form::new().part("file", Part::bytes(bytes).file_name(name.to_string()));
    app.http.post(app.url("/v1/ocr")).bearer_auth(token).multipart(form).send().await.unwrap()
}

/// PDF válido e vazio com `pages` páginas.
fn minimal_pdf(pages: usize) -> Vec<u8> {
    let mut objs: Vec<String> = vec!["<< /Type /Catalog /Pages 2 0 R >>".into()];
    let kids: String = (0..pages).map(|i| format!("{} 0 R ", 3 + i)).collect();
    objs.push(format!("<< /Type /Pages /Kids [{kids}] /Count {pages} >>"));
    for _ in 0..pages {
        objs.push("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".into());
    }
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = vec![];
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).bytes());
    }
    let xref = out.len();
    out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).bytes());
    for off in offsets {
        out.extend(format!("{off:010} 00000 n \n").bytes());
    }
    out.extend(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).bytes());
    out
}

#[tokio::test]
async fn ocr_happy_path_returns_text_and_hides_server_paths() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let r = upload(&app, &token, "../../relatório final.docx", minimal_docx("Olá mundo seguro")).await;
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["filename"], "relatório final.docx", "caminho do nome enviado deve ser removido");
    let docs = body["documents"].as_array().unwrap();
    assert!(!docs.is_empty());
    assert!(docs[0]["page_content"].as_str().unwrap().contains("Olá mundo seguro"));
    let all = body.to_string();
    assert!(!all.contains("input."), "nome interno do arquivo vazou: {all}");
    assert!(!all.contains("/tmp") && !all.contains("input.docx") && !all.contains("ocr-"), "vazou caminho temporário: {all}");
}

#[tokio::test]
async fn uppercase_extension_is_accepted() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let r = upload(&app, &token, "RELATÓRIO.DOCX", minimal_docx("Maiúsculas")).await;
    assert_eq!(r.status(), 200, "{}", r.text().await.unwrap());
}

#[tokio::test]
async fn ocr_requires_auth_and_post() {
    let app = TestApp::start().await;
    let form = Form::new().part("file", Part::bytes(minimal_docx("x")).file_name("a.docx"));
    assert_eq!(app.http.post(app.url("/v1/ocr")).multipart(form).send().await.unwrap().status(), 401);
}

#[tokio::test]
async fn disallowed_extensions_are_rejected() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    // HTML/XML/MD/JSON/áudio ficam de fora: são vetores de SSRF/XXE e processamento caro.
    for name in ["x.html", "x.htm", "x.xml", "x.md", "x.json", "x.svg", "x.mp3", "x.exe", "x.csv", "semextensao", "x.pdf.exe", "x."] {
        let r = upload(&app, &token, name, b"<html><img src=http://169.254.169.254/latest/meta-data/></html>".to_vec()).await;
        assert_eq!(r.status(), 422, "{name}");
    }
}

#[tokio::test]
async fn content_must_match_extension() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    for (name, bytes) in [
        ("fake.pdf", b"<html>nao sou pdf</html>".to_vec()),
        ("fake.png", b"%PDF-1.4 nao sou png".to_vec()),
        ("fake.docx", b"texto puro".to_vec()),
        ("fake.docx", minimal_pdf(1)),
    ] {
        let r = upload(&app, &token, name, bytes).await;
        assert_eq!(r.status(), 422, "{name}");
        assert!(r.text().await.unwrap().contains("não corresponde"), "{name}: rejeitado por outro motivo");
    }
    // ZIP válido, mas não é um DOCX (falta word/document.xml).
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut z = zip::ZipWriter::new(&mut buf);
        z.start_file("qualquer.txt", zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(b"oi").unwrap();
        z.finish().unwrap();
    }
    let r = upload(&app, &token, "x.docx", buf.into_inner()).await;
    assert_eq!(r.status(), 422);
    assert!(r.text().await.unwrap().contains("não corresponde"));
    // Vazio.
    assert!(upload(&app, &token, "x.pdf", vec![]).await.status().is_client_error());
}

#[tokio::test]
async fn pdf_page_limit_is_enforced() {
    let app = TestApp::start().await; // limite de 5 páginas no teste
    let (token, _) = app.tokens().await;
    let r = upload(&app, &token, "muitas.pdf", minimal_pdf(6)).await;
    assert_eq!(r.status(), 422);
    assert!(r.text().await.unwrap().contains("páginas"));
}

#[tokio::test]
async fn pixel_bomb_is_rejected_before_decoding() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    // Cabeçalho PNG de 30000x30000 (900 MP) — poucos bytes, memória enorme se decodificado.
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend(13u32.to_be_bytes());
    png.extend(b"IHDR");
    png.extend(30_000u32.to_be_bytes());
    png.extend(30_000u32.to_be_bytes());
    png.extend([8, 2, 0, 0, 0]);
    png.extend([0, 0, 0, 0]); // CRC (irrelevante para a checagem de cabeçalho)
    let r = upload(&app, &token, "bomba.png", png).await;
    assert_eq!(r.status(), 422);
    assert!(r.text().await.unwrap().contains("pixels"), "rejeitada por outro motivo");
}

#[tokio::test]
async fn zip_bomb_is_rejected_without_extracting() {
    let app = TestApp::start_with(|c| c.ocr_max_upload_bytes = 8 * 1024 * 1024).await;
    let (token, _) = app.tokens().await;
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut z = zip::ZipWriter::new(&mut buf);
        let o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        z.start_file("word/document.xml", o).unwrap();
        z.write_all(b"<w:document/>").unwrap();
        z.start_file("word/enorme.bin", o).unwrap();
        let zeros = vec![0u8; 1024 * 1024];
        for _ in 0..320 {
            z.write_all(&zeros).unwrap(); // 320 MiB descompactados, ~300 KB no disco
        }
        z.finish().unwrap();
    }
    let bytes = buf.into_inner();
    assert!(bytes.len() < 4 * 1024 * 1024, "a bomba precisa caber no limite de upload ({} bytes)", bytes.len());
    let started = Instant::now();
    let r = upload(&app, &token, "bomba.docx", bytes).await;
    assert_eq!(r.status(), 422);
    assert!(r.text().await.unwrap().contains("expande além do permitido"), "rejeitada por outro motivo");
    assert!(started.elapsed() < Duration::from_secs(10), "rejeição deve ser barata");
}

#[tokio::test]
async fn only_the_file_field_is_accepted() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let form = Form::new().text("comando", "rm -rf /").part("file", Part::bytes(minimal_docx("x")).file_name("a.docx"));
    let r = app.http.post(app.url("/v1/ocr")).bearer_auth(&token).multipart(form).send().await.unwrap();
    assert_eq!(r.status(), 400);

    let r = app.http.post(app.url("/v1/ocr")).bearer_auth(&token).multipart(Form::new()).send().await.unwrap();
    assert_eq!(r.status(), 400);

    let r = app.http.post(app.url("/v1/ocr")).bearer_auth(&token).header("content-type", "application/json").body("{}").send().await.unwrap();
    assert!(r.status().is_client_error());
}

#[tokio::test]
async fn one_ocr_at_a_time_per_user_and_global_capacity() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    let uid: uuid::Uuid = sqlx::query_scalar("SELECT id FROM users").fetch_one(&app.db).await.unwrap();

    // Usuário já tem um OCR em andamento => 429.
    let guard = app.state.try_user_guard(uid).expect("livre");
    let r = upload(&app, &token, "a.docx", minimal_docx("x")).await;
    assert_eq!(r.status(), 429);
    assert!(r.headers().contains_key("retry-after"));
    drop(guard);
    assert_eq!(upload(&app, &token, "a.docx", minimal_docx("x")).await.status(), 200);

    // Todas as vagas globais ocupadas => 503 (sem enfileirar para sempre).
    let held = app.state.ocr.slots.clone().acquire_many_owned(2).await.unwrap();
    let started = Instant::now();
    let r = upload(&app, &token, "a.docx", minimal_docx("x")).await;
    assert_eq!(r.status(), 503);
    assert!(r.headers().contains_key("retry-after"));
    assert!(started.elapsed() < Duration::from_secs(6));
    drop(held);
    assert_eq!(upload(&app, &token, "a.docx", minimal_docx("x")).await.status(), 200);
}

#[tokio::test]
async fn ocr_has_its_own_per_user_rate_limit() {
    let app = TestApp::start_with(|c| c.rate_ocr_per_min = 4).await; // rajada = 2
    let (token, _) = app.tokens().await;
    let mut statuses = vec![];
    for _ in 0..6 {
        statuses.push(upload(&app, &token, "a.docx", minimal_docx("x")).await.status().as_u16());
    }
    assert!(statuses.contains(&429), "{statuses:?}");
    assert_eq!(statuses[0], 200);
}

#[tokio::test]
async fn corrupt_office_file_gives_generic_error() {
    let app = TestApp::start().await;
    let (token, _) = app.tokens().await;
    // ZIP com o marcador certo, mas XML quebrado: o motor falha; a API não vaza o motivo.
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut z = zip::ZipWriter::new(&mut buf);
        z.start_file("word/document.xml", zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(b"<<<<nao e xml").unwrap();
        z.finish().unwrap();
    }
    let r = upload(&app, &token, "ruim.docx", buf.into_inner()).await;
    let status = r.status();
    let text = r.text().await.unwrap();
    assert!(status == 422 || status == 200, "{status} {text}");
    assert!(!text.contains("src/") && !text.contains("/home/") && !text.contains("panicked"), "{text}");
}

#[tokio::test]
async fn ocr_is_refused_under_memory_pressure() {
    // Marca d'água de 1%: qualquer máquina real está acima, então o OCR deve ser recusado.
    let app = TestApp::start_with(|c| c.ocr_memory_watermark_pct = 1).await;
    let (token, _) = app.tokens().await;
    let started = Instant::now();
    let r = upload(&app, &token, "a.docx", minimal_docx("x")).await;
    assert_eq!(r.status(), 503);
    assert!(r.headers().contains_key("retry-after"));
    assert!(started.elapsed() < Duration::from_secs(2), "recusa deve ser imediata, antes de ler o corpo");
}
