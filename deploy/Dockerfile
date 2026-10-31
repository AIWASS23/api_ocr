# syntax=docker/dockerfile:1.7
# Imagem de produção do OCR API (CPU). Base Debian trixie: o ONNX Runtime pré-compilado exige glibc >= 2.38. Para GPU, parta de uma base nvidia/cuda 13 e use
# `--features cuda` (ver api/README.md). Contexto de build: a RAIZ do repositório.

# ---------- compilação ----------
FROM rust:1.99-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY api ./api
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p api_ocr \
 && cp target/release/api_ocr /api_ocr

# ---------- modelos ONNX (layout, tabelas, OCR) ----------
# Digest permite PRs do Dependabot quando a imagem da mesma tag é reconstruída.
FROM debian:trixie-slim@sha256:f6b08b09a05725f9431502d3b5714f5c4576e82c1a84fa1debe0c24a7e0c3ec4 AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /models
# Script oficial na mesma tag do Docling, com hash fixado (cadeia de suprimentos).
ARG DL_URL=https://raw.githubusercontent.com/docling-project/docling.rs/v2.53.1/scripts/install/download_dependencies.sh
ARG DL_SHA256=d1fd9c10b524cf726ada3d5c846a1a8fee062c7285cedcef1ea7a3556fd932a4
RUN curl -fsSL "$DL_URL" -o dl.sh \
 && echo "$DL_SHA256  dl.sh" | sha256sum -c - \
 && sh dl.sh --no-asr

# ---------- dados do NLTK (estratégia de chunking `nltk`) ----------
# Estágio próprio: não invalida o cache do download dos modelos acima.
FROM debian:trixie-slim@sha256:f6b08b09a05725f9431502d3b5714f5c4576e82c1a84fa1debe0c24a7e0c3ec4 AS nltk
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates unzip \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /nltk
# punkt_tab num commit fixo do nltk_data, com hash fixado. Só o português (o idioma que a API usa).
ARG PUNKT_URL=https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip
ARG PUNKT_SHA256=e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106
RUN curl -fsSL "$PUNKT_URL" -o /tmp/punkt_tab.zip \
 && echo "$PUNKT_SHA256  /tmp/punkt_tab.zip" | sha256sum -c - \
 && mkdir tokenizers \
 && unzip -q /tmp/punkt_tab.zip 'punkt_tab/portuguese/*' -d tokenizers \
 && rm /tmp/punkt_tab.zip

# ---------- runtime ----------
FROM debian:trixie-slim@sha256:f6b08b09a05725f9431502d3b5714f5c4576e82c1a84fa1debe0c24a7e0c3ec4
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build  /api_ocr /app/api_ocr
COPY --from=models /models/.models /app/.models
# Um dos diretórios padrão do NLTK (sob /usr: legível pelo sandbox, sem configuração).
COPY --from=nltk /nltk /usr/share/nltk_data

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
