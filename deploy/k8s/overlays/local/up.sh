#!/usr/bin/env bash
# Sobe o ambiente local. Gera segredos aleatórios e um certificado autoassinado na primeira vez
# (em ./secrets, ignorado pelo git) e aplica o overlay.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
s="$here/secrets"
mkdir -p "$s"
chmod 700 "$s"
rand() { head -c 48 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=\n'; }

if [ ! -f "$s/db.env" ]; then
  pg="$(rand)"; rd="$(rand)"
  umask 077
  printf 'POSTGRES_PASSWORD=%s\nREDIS_PASSWORD=%s\n' "$pg" "$rd" > "$s/db.env"
  printf 'JWT_SECRET=%s\nPASSWORD_PEPPER=%s\nDATABASE_URL=postgres://api_ocr:%s@postgres:5432/api_ocr\nRATE_LIMIT_REDIS_URL=redis://:%s@redis:6379\n' \
    "$(rand)" "$(rand)" "$pg" "$rd" > "$s/app.env"
  echo "segredos gerados em $s"
fi
if [ ! -f "$s/tls.crt" ]; then
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 30 \
    -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost" \
    -keyout "$s/tls.key" -out "$s/tls.crt" 2>/dev/null
  echo "certificado autoassinado gerado (CN=localhost)"
fi

kubectl apply -k "$here"
kubectl -n api-ocr rollout status statefulset/postgres --timeout=180s
kubectl -n api-ocr rollout status deployment/redis --timeout=120s
kubectl -n api-ocr rollout status deployment/api-ocr --timeout=300s
echo "pronto: https://localhost:18443  (certificado autoassinado)"
