# Kubernetes

Manifests com Kustomize: `base/` (comum) e dois overlays.

| Overlay | Banco e Redis | TLS | Segredos |
|---|---|---|---|
| `overlays/local` | Postgres (StatefulSet) e Redis no cluster | autoassinado | gerados pelo `up.sh` em `secrets/` (ignorado pelo git) |
| `overlays/production` | gerenciados, fora do cluster, com TLS | cert-manager | vindos de fora (External Secrets, Sealed Secrets, CSI) |

Requisitos do cluster:

- **Gateway API** com o **Envoy Gateway** (o ingress-nginx foi descontinuado em março de 2026). Os
  objetos são Gateway API padrão (`Gateway`, `HTTPRoute`); só `EnvoyProxy` e `ClientTrafficPolicy`,
  no overlay de produção, são específicos do Envoy Gateway.
- **CNI que aplique NetworkPolicy** (Calico, Cilium, o kube-router embutido no k3s...). Sem isso as
  políticas são ignoradas em silêncio, e o `TRUST_PROXY=true` passa a aceitar `X-Forwarded-For`
  forjado vindo de qualquer pod.
- **Nós com Landlock** (kernel ≥ 6.7 para a parte de rede). O pod usa `SANDBOX=required` e se recusa
  a subir sem o sandbox completo.

## Ambiente local (k3d)

```bash
k3d cluster create api-ocr --agents 2 --k3s-arg "--disable=traefik@server:0" \
    -p "18080:80@loadbalancer" -p "18443:443@loadbalancer" --wait
helm install eg oci://docker.io/envoyproxy/gateway-helm --version v1.9.2 \
    -n envoy-gateway-system --create-namespace --wait
docker build -f deploy/Dockerfile -t api-ocr:k8s-test .
k3d image import api-ocr:k8s-test -c api-ocr
deploy/k8s/overlays/local/up.sh                       # https://localhost:18443

kubectl -n api-ocr exec -i deploy/api-ocr -- /app/api_ocr create-user --email voce@exemplo.com <<< 'sua-senha-longa'
kubectl -n api-ocr exec deploy/api-ocr -- /app/api_ocr create-client --id cli --name CLI \
    --redirect-uri http://127.0.0.1:8765/callback
```

Os comandos do operador (`mfa-enable`, `revoke-sessions`, `unlock-user`...) rodam do mesmo jeito,
com `kubectl exec`.

## Produção

1. Troque todo valor marcado com `TROQUE` em `overlays/production/`: imagem (de preferência por
   digest), `ISSUER`, hostname do Gateway, `ClusterIssuer` do cert-manager e as faixas de IP do
   banco/Redis na NetworkPolicy de saída.
2. Crie o Secret `api-ocr-secrets` (veja `external-secret.example.yaml`). A `DATABASE_URL` precisa de
   `sslmode=verify-full` (ou `require`): sem TLS, a API não sobe.
3. `kubectl apply -k deploy/k8s/overlays/production`

O `EnvoyProxy` usa `externalTrafficPolicy: Local` para preservar o IP do cliente atrás do balanceador
da nuvem. Sem isso, todos os clientes chegariam com o IP de um nó, e o limite de taxa, o bloqueio de
login e a auditoria passariam a valer por nó em vez de por cliente.

## O que os manifests garantem

| Item | Como |
|---|---|
| Pods sem privilégio | Namespace com Pod Security `restricted` (enforce): não-root, sem capacidades, `seccompProfile: RuntimeDefault`, raiz somente leitura, `allowPrivilegeEscalation: false`, sem token de ServiceAccount |
| Só o Gateway fala com a API | NetworkPolicy `default-deny` (entrada e saída) + exceções: Gateway → API:8080, API → Postgres/Redis, DNS. Postgres e Redis só aceitam a API |
| Só as rotas da API são públicas | `HTTPRoute` publica `/oauth/`, `/auth/`, `/v1/` e `/.well-known/oauth-authorization-server`; `/docs`, `/openapi.json`, `/healthz` e `/readyz` dão 404 na borda. HTTP só redireciona para HTTPS |
| Disponibilidade | 2+ réplicas em nós diferentes, PDB (`minAvailable: 1`), HPA por CPU (2–6), rollout sem indisponibilidade (`maxUnavailable: 0`), `preStop` de 5 s, encerramento gracioso |
| Uploads não tocam o disco do nó | `/tmp` é `emptyDir` em memória |
| Memória | O vigia da API mata o maior worker de OCR a 85% do `limits.memory` (medido a cada 10 ms). Sem ele, o OOM do kernel (cgroup v2, `memory.oom.group=1`) derrubaria o contêiner inteiro |

## Testado (k3d, k3s v1.35, Envoy Gateway v1.9.2)

- Subida sem nenhum aviso do Pod Security; Landlock e seccomp ativos dentro dos pods; warm-up em 4 s.
- Fluxo completo pelo Gateway: login com PKCE, token com `kid`, `/v1/me` e OCR de PDF escaneado
  (~2,6 s); rotas não publicadas dão 404; HTTP dá 301 para HTTPS.
- NetworkPolicy: de outro namespace, API, Postgres e Redis recusam conexão; do namespace do Envoy,
  só a API aceita.
- `X-Forwarded-For` forjado pelo cliente é ignorado (a auditoria grava o IP que o Envoy viu).
- Overlay de produção: os 17 objetos passam na validação do servidor (`--dry-run=server`).
- Memória: com `limits.memory` reduzido para 1,2 GiB (menos do que um job usa), 10 de 10 subidas
  tiveram o worker morto pelo vigia e **nenhum** contêiner sofreu OOM; um OCR nessas condições
  responde 422 em ~1 s e registra `ocr_worker_memory_killed`, sem reiniciar o pod.

**Não testado:** cert-manager emitindo certificado real, banco gerenciado com TLS, preservação do IP
real atrás de um balanceador de nuvem (no k3d há dois NATs antes do Envoy) e GPU no cluster.
