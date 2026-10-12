#!/usr/bin/env bash
# Certificado AUTOASSINADO só para testes locais. Em produção use Let's Encrypt/ACM etc.
set -euo pipefail
host="${1:-localhost}"
dir="$(cd "$(dirname "$0")" && pwd)/certs"
mkdir -p "$dir"
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 30 \
  -subj "/CN=$host" -addext "subjectAltName=DNS:$host" \
  -keyout "$dir/privkey.pem" -out "$dir/fullchain.pem" 2>/dev/null
# Só para teste: o nginx-unprivileged (uid 101) precisa ler a chave. Em produção, monte o
# certificado como segredo legível apenas pelo usuário do nginx.
chmod 644 "$dir/privkey.pem"
echo "gerado em $dir (CN=$host)"
