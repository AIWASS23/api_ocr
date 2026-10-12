# syntax=docker/dockerfile:1.7
# Imagem de produção do OCR API (CPU). Base Debian trixie: o ONNX Runtime pré-compilado exige glibc >= 2.38. Para GPU, parta de uma base nvidia/cuda 13 e use
# `--features cuda` (ver api/README.md). Contexto de build: a RAIZ do repositório.

# ---------- compilação ----------
FROM rust:1.95-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY api ./api
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p api_ocr \
 && cp target/release/api_ocr /api_ocr

# ---------- modelos ONNX (layout, tabelas, OCR) ----------
FROM debian:trixie-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /models
# Script oficial na tag v2.15.0, com hash fixado (cadeia de suprimentos).
ARG DL_URL=https://raw.githubusercontent.com/docling-project/docling.rs/v2.15.0/scripts/install/download_dependencies.sh
ARG DL_SHA256=78425527c90826c49ed38e859d240dab5f8469d941dff5ba3c360ca6bb2f4273
RUN curl -fsSL "$DL_URL" -o dl.sh \
 && echo "$DL_SHA256  dl.sh" | sha256sum -c - \
 && sh dl.sh --no-asr

# ---------- runtime ----------
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build  /api_ocr /app/api_ocr
COPY --from=models /models/.models /app/.models

# Raiz somente leitura: tudo que o app grava vai para /tmp (tmpfs).
ENV BIND_ADDR=0.0.0.0:8080 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    SANDBOX=required \
    RUST_LOG=info,sqlx=warn
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 CMD ["/app/api_ocr", "healthcheck"]
ENTRYPOINT ["/app/api_ocr"]
CMD ["serve"]
