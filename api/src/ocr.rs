//! `POST /v1/data_preparation`: recebe UM arquivo (multipart) e devolve o texto/tabelas extraídos.
//!
//! Defesas (o OCR é a parte mais cara e a mais exposta do servidor):
//! * limites de taxa por usuário, 1 OCR por vez por usuário e N globais;
//! * o corpo só é lido DEPOIS de reservar as vagas (flood de upload não gasta disco/banda);
//! * tamanho máximo, tempo máximo de upload e de processamento;
//! * extensão em lista de permitidos + conferência de assinatura (magic bytes);
//! * limites anti-bomba: páginas de PDF, pixels de imagem, entradas/expansão de ZIP;
//! * arquivo temporário com nome fixo (o nome enviado nunca vira caminho) e removido ao fim;
//! * TODA leitura do conteúdo (inspeção e conversão) acontece num processo worker separado,
//!   sem segredos, sem rede, com seccomp e limites de memória/CPU (ver `worker.rs`): um bug
//!   explorável num parser não alcança a memória da API;
//! * erros internos nunca chegam ao cliente.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::{Multipart, State};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::io::AsyncWriteExt;

use crate::auth::Auth;
use crate::error::ApiError;
use crate::models::{ErrorBody, OcrDocument, OcrResponse, OcrUpload};
use crate::state::AppState;
use crate::worker::{self, JobKind, WorkerFailure};

const UPLOAD_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_PIXELS: u64 = 100_000_000;
const MAX_ZIP_ENTRIES: usize = 10_000;
const MAX_ZIP_UNCOMPRESSED: u64 = 300 * 1024 * 1024;
const MAX_ZIP_RATIO: u64 = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Pdf,
    Image,
    Docx,
    Pptx,
    Xlsx,
}

/// Formatos aceitos pela API (HTML/XML/MD/JSON/áudio ficam de fora de propósito).
pub(crate) fn kind_for_ext(ext: &str) -> Option<Kind> {
    Some(match ext {
        "pdf" => Kind::Pdf,
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "webp" => Kind::Image,
        "docx" => Kind::Docx,
        "pptx" => Kind::Pptx,
        "xlsx" => Kind::Xlsx,
        _ => return None,
    })
}

/// Só o nome-base, sem caminho, com caracteres seguros e tamanho limitado.
fn sanitize_filename(raw: &str) -> String {
    let base = raw.rsplit(['/', '\\']).next().unwrap_or("");
    let cleaned: String = base
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '_' | ' '))
        .take(100)
        .collect();
    let cleaned = cleaned.trim_matches(['.', ' ']).to_string();
    if cleaned.is_empty() { "documento".to_string() } else { cleaned }
}

pub(crate) fn extension(name: &str) -> Option<String> {
    let (_, ext) = name.rsplit_once('.')?;
    Some(ext.to_ascii_lowercase())
}

fn magic_matches(kind: Kind, head: &[u8]) -> bool {
    match kind {
        Kind::Pdf => head.starts_with(b"%PDF-"),
        Kind::Docx | Kind::Pptx | Kind::Xlsx => head.starts_with(b"PK\x03\x04"),
        Kind::Image => {
            head.starts_with(b"\x89PNG\r\n\x1a\n")
                || head.starts_with(&[0xFF, 0xD8, 0xFF])
                || head.starts_with(b"II*\0")
                || head.starts_with(b"MM\0*")
                || head.starts_with(b"BM")
                || (head.len() >= 12 && &head[..4] == b"RIFF" && &head[8..12] == b"WEBP")
        }
    }
}

/// Motivo de recusa de um arquivo (422). O worker devolve só o código; a mensagem é desta
/// tabela fixa, então nem um worker comprometido consegue pôr texto arbitrário na resposta.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    Unreadable,
    ContentMismatch,
    PdfInvalid,
    PdfEmpty,
    TooManyPages,
    ImageInvalid,
    ImageTooLarge,
    OfficeInvalid,
    OfficeEntries,
    OfficeNames,
    OfficePaths,
    OfficeExpansion,
    UnsupportedFormat,
    ConversionFailed,
}

impl Rejection {
    pub fn message(self) -> &'static str {
        match self {
            Rejection::Unreadable => "arquivo ilegível",
            Rejection::ContentMismatch => "o conteúdo do arquivo não corresponde à extensão",
            Rejection::PdfInvalid => "PDF inválido ou protegido",
            Rejection::PdfEmpty => "PDF sem páginas",
            Rejection::TooManyPages => "o documento excede o número máximo de páginas",
            Rejection::ImageInvalid => "imagem inválida",
            Rejection::ImageTooLarge => "imagem grande demais (pixels)",
            Rejection::OfficeInvalid => "arquivo Office inválido",
            Rejection::OfficeEntries => "arquivo Office com número de entradas inválido",
            Rejection::OfficeNames => "arquivo Office com nomes inválidos",
            Rejection::OfficePaths => "arquivo Office com caminhos inválidos",
            Rejection::OfficeExpansion => "arquivo Office expande além do permitido",
            Rejection::UnsupportedFormat => "formato não suportado",
            Rejection::ConversionFailed => "não foi possível processar o documento",
        }
    }
}

/// Inspeção barata e síncrona do arquivo já gravado (roda no worker). `Err` => 422.
fn inspect(path: &Path, kind: Kind, max_pages: usize) -> Result<(), Rejection> {
    let mut head = [0u8; 16];
    let n = std::fs::File::open(path)
        .and_then(|mut f| f.read(&mut head))
        .map_err(|_| Rejection::Unreadable)?;
    if !magic_matches(kind, &head[..n]) {
        return Err(Rejection::ContentMismatch);
    }

    match kind {
        Kind::Pdf => {
            let pages = data_preparation::pdf_page_count(path).ok_or(Rejection::PdfInvalid)?;
            if pages == 0 {
                return Err(Rejection::PdfEmpty);
            }
            if pages > max_pages {
                return Err(Rejection::TooManyPages);
            }
        }
        Kind::Image => {
            let dim = imagesize::size(path).map_err(|_| Rejection::ImageInvalid)?;
            if (dim.width as u64).saturating_mul(dim.height as u64) > MAX_PIXELS {
                return Err(Rejection::ImageTooLarge);
            }
        }
        Kind::Docx | Kind::Pptx | Kind::Xlsx => inspect_zip(path, kind)?,
    }
    Ok(())
}

/// Lê só o diretório central do ZIP (sem descompactar) para barrar zip bombs.
fn inspect_zip(path: &Path, kind: Kind) -> Result<(), Rejection> {
    let file = std::fs::File::open(path).map_err(|_| Rejection::Unreadable)?;
    let mut zip = zip::ZipArchive::new(file).map_err(|_| Rejection::OfficeInvalid)?;
    if zip.is_empty() || zip.len() > MAX_ZIP_ENTRIES {
        return Err(Rejection::OfficeEntries);
    }
    let (mut total, mut compressed) = (0u64, 0u64);
    for i in 0..zip.len() {
        let e = zip.by_index_raw(i).map_err(|_| Rejection::OfficeInvalid)?;
        let name = e.name().map_err(|_| Rejection::OfficeNames)?;
        if name.starts_with('/') || name.split(['/', '\\']).any(|p| p == "..") {
            return Err(Rejection::OfficePaths);
        }
        total = total.saturating_add(e.size());
        compressed = compressed.saturating_add(e.compressed_size());
    }
    if total > MAX_ZIP_UNCOMPRESSED || total > compressed.max(1).saturating_mul(MAX_ZIP_RATIO) {
        return Err(Rejection::OfficeExpansion);
    }
    let marker = match kind {
        Kind::Docx => "word/document.xml",
        Kind::Pptx => "ppt/presentation.xml",
        _ => "xl/workbook.xml",
    };
    if !zip.file_names().any(|n| n.as_deref().is_ok_and(|n| n == marker)) {
        return Err(Rejection::ContentMismatch);
    }
    Ok(())
}

/// Documento extraído, como trafega do worker para a API.
#[derive(Debug, Serialize, Deserialize)]
pub struct WireDocument {
    pub page_content: String,
    pub metadata: Map<String, Value>,
}

/// Duração das fases DENTRO do worker, em milissegundos (só vão para o log da API).
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Timings {
    /// rlimits + Landlock + seccomp.
    pub confine_ms: u64,
    /// Assinatura e limites anti-bomba (páginas, pixels, ZIP).
    pub inspect_ms: u64,
    /// Montagem do conversor.
    pub engine_ms: u64,
    /// Conversão, incluindo a carga dos modelos ONNX (o docling os carrega no 1º documento).
    pub convert_ms: u64,
}

fn ms(d: Duration) -> u64 {
    d.as_millis().try_into().unwrap_or(u64::MAX)
}

/// Resposta do worker (uma linha JSON no stdout).
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Reply {
    Ok {
        documents: Vec<WireDocument>,
        #[serde(default)]
        timings: Timings,
    },
    Rejected { reason: Rejection },
}

/// Executa um job inteiro DENTRO do worker: inspeção anti-bomba + conversão.
/// `Err` = falha interna (o worker sai com código de erro e a API responde genericamente).
pub fn execute_job(path: &Path, max_pages: usize, timeout: Duration) -> Result<Reply, String> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let Some(kind) = extension(name).as_deref().and_then(kind_for_ext) else {
        return Ok(Reply::Rejected { reason: Rejection::UnsupportedFormat });
    };
    let mut timings = Timings::default();
    let t = Instant::now();
    if let Err(reason) = inspect(path, kind, max_pages) {
        return Ok(Reply::Rejected { reason });
    }
    timings.inspect_ms = ms(t.elapsed());
    let t = Instant::now();
    let engine = crate::ocr_engine(timeout, max_pages).map_err(|e| e.to_string())?;
    timings.engine_ms = ms(t.elapsed());
    let t = Instant::now();
    let (docs, errors) = engine.process_files(path, "txt").map_err(|e| e.to_string())?;
    timings.convert_ms = ms(t.elapsed());
    if docs.is_empty()
        && let Some(e) = errors.first() {
            // Vai para o stderr do worker (a API registra um trecho); nunca para o cliente.
            eprintln!("conversão falhou: {}", e.error);
            return Ok(Reply::Rejected { reason: Rejection::ConversionFailed });
        }
    Ok(Reply::Ok {
        documents: docs.into_iter().map(|d| WireDocument { page_content: d.page_content, metadata: d.metadata }).collect(),
        timings,
    })
}

fn multipart_error(e: axum::extract::multipart::MultipartError) -> ApiError {
    if e.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::PayloadTooLarge
    } else {
        ApiError::BadRequest("corpo multipart inválido")
    }
}

/// Grava o campo em `dest`, parando ao passar de `max` bytes.
async fn save_field(mut field: axum::extract::multipart::Field<'_>, dest: &Path, max: usize) -> Result<(), ApiError> {
    let mut out = tokio::fs::File::create(dest).await.map_err(|e| ApiError::internal("criar temporário", e))?;
    let mut written = 0usize;
    while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
        written += chunk.len();
        if written > max {
            return Err(ApiError::PayloadTooLarge);
        }
        out.write_all(&chunk).await.map_err(|e| ApiError::internal("gravar temporário", e))?;
    }
    out.flush().await.map_err(|e| ApiError::internal("gravar temporário", e))?;
    if written == 0 {
        return Err(ApiError::BadRequest("arquivo vazio"));
    }
    Ok(())
}

/// Extrai texto e tabelas de um documento.
///
/// Envie **um** arquivo no campo multipart `file`. O corpo só é lido depois de reservar as vagas de
/// processamento; se o servidor estiver ocupado (ou com pouca memória) responde 503 com `Retry-After`.
/// Cada usuário processa 1 documento por vez.
#[utoipa::path(
    post, path = "/v1/data_preparation", tag = "Preparação de dados",
    security(("bearer" = []), ("oauth2" = ["ocr"])),
    request_body(content = OcrUpload, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Texto e linhas de tabela extraídos", body = OcrResponse),
        (status = 400, description = "Multipart inválido ou campo diferente de `file`", body = ErrorBody),
        (status = 401, description = "Token ausente, inválido ou expirado", body = ErrorBody),
        (status = 403, description = "Falta o escopo `ocr`", body = ErrorBody),
        (status = 413, description = "Arquivo acima do limite", body = ErrorBody),
        (status = 422, description = "Formato não suportado, conteúdo não confere com a extensão, páginas/pixels/expansão acima do limite ou documento ilegível", body = ErrorBody),
        (status = 429, description = "Limite de taxa ou OCR já em andamento para este usuário", body = ErrorBody),
        (status = 503, description = "Servidor ocupado, sem memória ou tempo esgotado", body = ErrorBody),
    )
)]
pub async fn ocr(State(st): State<AppState>, auth: Auth, mut multipart: Multipart) -> Result<Json<OcrResponse>, ApiError> {
    let started = Instant::now();
    auth.require_scope("ocr")?;
    let user = auth.user_id()?;

    // 1) Limites ANTES de ler o corpo.
    st.limiters.check_ocr(user).await.map_err(ApiError::TooManyRequests)?;
    if st.memory.is_under_pressure() {
        tracing::warn!("OCR recusado: memória acima do limite configurado");
        return Err(ApiError::Unavailable(Duration::from_secs(15)));
    }
    let guard = st
        .try_user_guard(user)
        .ok_or(ApiError::TooManyRequests(Duration::from_secs(5)))?;
    let permit = match tokio::time::timeout(Duration::from_secs(2), st.ocr.slots.clone().acquire_owned()).await {
        Ok(Ok(p)) => p,
        _ => return Err(ApiError::Unavailable(Duration::from_secs(10))),
    };

    let admit_ms = ms(started.elapsed());

    // 2) Recebe exatamente um campo `file`.
    let t_upload = Instant::now();
    let tmp = tempfile::Builder::new()
        .prefix("ocr-")
        .tempdir()
        .map_err(|e| ApiError::internal("criar diretório temporário", e))?;

    let (display_name, path): (String, PathBuf) = {
        let upload = async {
            let field = multipart
                .next_field()
                .await
                .map_err(multipart_error)?
                .ok_or(ApiError::BadRequest("campo 'file' ausente"))?;
            if field.name() != Some("file") {
                return Err(ApiError::BadRequest("o único campo aceito é 'file'"));
            }
            let display = sanitize_filename(field.file_name().unwrap_or(""));
            let ext = extension(&display).ok_or(ApiError::Unprocessable("o arquivo precisa ter extensão"))?;
            kind_for_ext(&ext).ok_or(ApiError::Unprocessable(Rejection::UnsupportedFormat.message()))?;
            // `display` já é só um nome-base seguro (sem barras, sem pontos nas pontas) e a
            // extensão foi validada acima; o arquivo fica num diretório temporário exclusivo.
            // Usar o nome real faz o motor citar o arquivo certo nos metadados e no texto.
            debug_assert!(!display.contains(['/', '\\']) && display.to_ascii_lowercase().ends_with(&ext));
            let path = tmp.path().join(&display);
            save_field(field, &path, st.cfg.ocr_max_upload_bytes).await?;
            Ok::<_, ApiError>((display, path))
        };
        tokio::time::timeout(UPLOAD_TIMEOUT, upload)
            .await
            .map_err(|_| ApiError::BadRequest("tempo de upload esgotado"))??
    };
    // Qualquer campo extra é ignorado (não é lido nem armazenado).
    drop(multipart);
    let upload_ms = ms(t_upload.elapsed());
    let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());

    // 3) Processa num processo worker isolado. Ao estourar o prazo (ou o cliente desistir) o
    //    processo é morto de verdade; as vagas só são liberadas depois disso.
    let t_worker = Instant::now();
    let outcome = worker::run_job(&st.cfg, tmp.path(), &path).await;
    let worker_ms = ms(t_worker.elapsed());
    drop((permit, guard, tmp)); // apaga o diretório do job

    let docs = match outcome {
        Ok(Reply::Ok { documents, timings: w }) => {
            // Tempo de cada fase (sem conteúdo nem nome do arquivo). `worker_spawn_ms` = criar o
            // processo, carregar o binário e devolver a resposta: o custo do isolamento.
            let inside = w.confine_ms + w.inspect_ms + w.engine_ms + w.convert_ms;
            tracing::info!(
                bytes,
                documents = documents.len(),
                admit_ms,
                upload_ms,
                worker_ms,
                worker_spawn_ms = worker_ms.saturating_sub(inside),
                confine_ms = w.confine_ms,
                inspect_ms = w.inspect_ms,
                engine_ms = w.engine_ms,
                convert_ms = w.convert_ms,
                total_ms = ms(started.elapsed()),
                "OCR concluído"
            );
            documents
        }
        Ok(Reply::Rejected { reason }) => {
            if reason == Rejection::ConversionFailed {
                tracing::warn!("conversão falhou");
            }
            return Err(ApiError::Unprocessable(reason.message()));
        }
        Err(WorkerFailure::Timeout) => return Err(ApiError::Unavailable(Duration::from_secs(30))),
        Err(failure) => {
            crate::security::event(&st, failure.event_kind(JobKind::Ocr), Some(user), None, None, Some(&failure.to_string())).await;
            return Err(ApiError::Unprocessable(Rejection::ConversionFailed.message()));
        }
    };

    let documents: Vec<OcrDocument> = docs
        .into_iter()
        .map(|d| {
            let mut metadata: Map<String, Value> = d.metadata;
            // Nunca expõe o caminho temporário do servidor.
            metadata.insert("source".into(), json!(display_name));
            OcrDocument { page_content: d.page_content, metadata }
        })
        .collect();

    Ok(Json(OcrResponse { filename: display_name, count: documents.len(), documents }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filenames_are_sanitized() {
        assert_eq!(sanitize_filename("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_filename("C:\\x\\relatório final.pdf"), "relatório final.pdf");
        assert_eq!(sanitize_filename("a\0b<script>.pdf"), "abscript.pdf");
        assert_eq!(sanitize_filename("...."), "documento");
        assert_eq!(sanitize_filename(&"a".repeat(500)).len(), 100);
    }

    #[test]
    fn only_allowed_extensions() {
        for ok in ["pdf", "png", "jpg", "docx", "pptx", "xlsx", "webp", "tiff"] {
            assert!(kind_for_ext(ok).is_some(), "{ok}");
        }
        for bad in ["html", "htm", "xml", "json", "md", "svg", "mp3", "exe", "csv", "", "pdf.exe"] {
            assert!(kind_for_ext(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn magic_bytes() {
        assert!(magic_matches(Kind::Pdf, b"%PDF-1.7"));
        assert!(!magic_matches(Kind::Pdf, b"<html><body>"));
        assert!(magic_matches(Kind::Docx, b"PK\x03\x04rest"));
        assert!(!magic_matches(Kind::Image, b"%PDF-1.7"));
        assert!(magic_matches(Kind::Image, b"\x89PNG\r\n\x1a\nxxxx"));
    }
}
