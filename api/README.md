# OCR API

API HTTP em Rust (axum + hyper) que expõe a preparação de dados (`src/data_preparation`, OCR com
docling.rs) e o chunking (`src/chunking`, 33 estratégias) atrás de um servidor OAuth2 próprio: **Authorization Code + PKCE (S256)**, **access token JWT curto**,
**refresh token com rotação e detecção de reuso**, rotas protegidas e PostgreSQL via `sqlx`.

## Subir em desenvolvimento

```bash
# 1) Postgres dedicado (127.0.0.1:55432). A senha fica em deploy/.env (ignorado pelo git).
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d

# 2) Configuração
cp api/.env.example api/.env        # preencha; gere segredos com `api_ocr gen-secret`
set -a; . api/.env; set +a

# 3) Build, banco, usuário e cliente
cargo build --release -p api_ocr
./target/release/api_ocr migrate
printf 'sua-senha-longa\n' | ./target/release/api_ocr create-user --email voce@exemplo.com
./target/release/api_ocr create-client --id ocr-cli --name "OCR CLI" \
    --redirect-uri http://127.0.0.1:8765/callback

# 4) Servidor. Rode da RAIZ do repositório: os modelos ONNX são lidos de ./.models
./target/release/api_ocr serve
```

GPU: compile com `--features cuda` e use `source scripts/cuda-env.sh` antes de subir.

## Endpoints

| Método e rota | Auth | Descrição |
|---|---|---|
| `GET /oauth/authorize` | — | Página de login (CSRF, CSP estrita). Parâmetros: `response_type=code`, `client_id`, `redirect_uri`, `scope`, `state`, `code_challenge`, `code_challenge_method=S256` |
| `POST /oauth/authorize` | cookie CSRF | Valida credenciais e redireciona com `code`, `state` e `iss` |
| `POST /oauth/token` | — | `grant_type=authorization_code` (com `code_verifier`) ou `refresh_token` |
| `POST /oauth/revoke` | — | Revoga a família de um refresh token (RFC 7009) |
| `GET /.well-known/oauth-authorization-server` | — | Metadados (RFC 8414) |
| `POST /auth/register` | — | Cadastro (`{email, password}`); só com `ALLOW_REGISTRATION=true` |
| `GET /v1/me` | Bearer | Dados do usuário autenticado |
| `POST /v1/data_preparation` | Bearer, escopo `ocr` | Multipart com **um** campo `file` (pdf, png, jpg, tif, bmp, webp, docx, pptx, xlsx); devolve `documents` (`page_content` + `metadata`) |
| `GET /v1/chunk/strategies` | Bearer, escopo `ocr` | Estratégias de chunking: apelidos, disponibilidade neste servidor, unidade e intervalo do `chunk_size`, padrão e máximo do `chunk_overlap` (ou `null`), limites |
| `POST /v1/chunk` | Bearer, escopo `ocr` | JSON `{strategy, chunk_size?, chunk_overlap?, text}` ou `{..., documents}` (a saída do `/v1/data_preparation`); devolve os chunks com metadados |
| `GET /healthz`, `GET /readyz` | — | Liveness / readiness (checa o banco) |
| `GET /docs`, `GET /openapi.json` | — | **Swagger UI** e OpenAPI 3.1 (só com `ENABLE_DOCS=true`; veja abaixo) |

Exemplo de extração seguida de chunking:

```bash
curl -H "Authorization: Bearer $ACCESS_TOKEN" -F "file=@relatorio.pdf" \
     http://localhost:8080/v1/data_preparation > extraido.json

# Os `documents` extraídos vão direto para o chunking (os metadados de cada um seguem para os chunks).
jq '{strategy: "markdown_header_contextual", chunk_size: 800, chunk_overlap: 80, documents: .documents}' extraido.json |
  curl -H "Authorization: Bearer $ACCESS_TOKEN" -H "Content-Type: application/json" --data-binary @- \
       http://localhost:8080/v1/chunk

# Ou um texto qualquer:
curl -H "Authorization: Bearer $ACCESS_TOKEN" -H "Content-Type: application/json" \
     -d '{"strategy": "sentence", "chunk_size": 500, "text": "Primeira frase. Segunda frase..."}' \
     http://localhost:8080/v1/chunk
```

### Chunking

- **Parâmetros:** só `strategy` (nome ou apelido), `chunk_size` e `chunk_overlap`. Padrões: 1000 caracteres
  (512 tokens nas estratégias medidas em tokens) e sobreposição de 10% do `chunk_size`; a sobreposição vai
  até a metade do `chunk_size`. Numa estratégia sem sobreposição (`docling_hybrid`), um `chunk_overlap`
  maior que 0 é recusado (422). As opções internas das estratégias (modelos, URLs, caminhos) não são
  aceitas do cliente: campos desconhecidos dão 400.
- **Disponibilidade:** 26 estratégias funcionam sem nada a mais. `token`, `semchunk`,
  `token_sentence_transformer` e `docling_hybrid` precisam de um tokenizador local (`CHUNK_TOKENIZER`).
  `semantic`, `late_chunking` e `contextual_retrieval` precisam de modelo de embeddings ou LLM e aparecem
  como indisponíveis. `GET /v1/chunk/strategies` mostra o estado de cada uma neste servidor.
- **Fixos do servidor:** `nltk` e `spacy` em português; `code` = Python; `token_tiktoken` trata tokens
  especiais (`<|endoftext|>`) como texto; `parent_child` devolve o pai em `metadata.parent_content`.
- **Isolamento:** a divisão roda num worker igual ao do OCR (ambiente zerado, Landlock, seccomp, sem rede,
  prazo de `CHUNK_TIMEOUT_SECS` e teto de memória). Texto patológico (HTML/JSON aninhados a ponto de
  estourar a pilha ou deixar o parser HTML quadrático) derruba ou trava só o worker: a API responde 422/503
  e segue de pé. Antes do worker, a API valida o pedido e recusa o que geraria chunks demais.

## Swagger / OpenAPI

```bash
ENABLE_DOCS=true ./target/release/api_ocr serve     # http://localhost:8080/docs
```

- A UI e o JSON (`/openapi.json`) **só existem com `ENABLE_DOCS=true`**; desligado, respondem 404.
- O Swagger UI vai **dentro do binário** (sem CDN). A CSP dele é isolada nestas rotas
  (`script-src 'self'`, sem scripts inline); o resto da API mantém `default-src 'none'`.
- O botão **Authorize** faz Authorization Code + PKCE de verdade. Ao ligar as docs, o cliente `swagger-ui`
  é registrado sozinho, com redirect apenas para `ISSUER/docs/oauth2-redirect.html`.
- As respostas de `/v1/me`, `/v1/data_preparation`, `/v1/chunk`, token etc. usam os **mesmos tipos Rust** que geram o schema, e
  os testes (`tests/docs.rs`) conferem que as rotas documentadas são exatamente as reais, que todo `$ref`
  resolve e que os campos das respostas reais batem com o schema.
- Atrás do nginx de produção, `/docs` fica **bloqueado por padrão** (bloco comentado no template);
  se for expor, restrinja por IP/VPN.

## Defesas implementadas (e o teste que prova cada uma)

| Ameaça | Defesa | Teste |
|---|---|---|
| **SQL injection** | 100% `sqlx` com `.bind()`. O `sqlx` 0.9 **não compila** SQL dinâmico (só literais `&'static str`); `grep AssertSqlSafe api/src` retorna vazio. Entrada validada por lista de permitidos. `statement_timeout` (10 s) e `lock_timeout` (5 s) no pool. | `sql_injection_in_every_input_is_inert` (10 payloads em todos os campos; confere que o banco não mudou e que `pg_sleep` não rodou) |
| **Token forjado** | HS256 com algoritmo **fixo** (rejeita `alg=none`, troca de algoritmo, payload adulterado, outro segredo); `iss`, `aud`, `exp`, `nbf` obrigatórios | `forged_and_malformed_tokens_are_rejected` |
| **Roubo de authorization code** | PKCE S256 obrigatório (sem `plain`), uso único atômico, TTL de 60 s, replay revoga os tokens emitidos | `pkce_is_enforced`, `authorization_code_cannot_be_replayed...` |
| **Roubo de refresh token** | Opaco (256 bits), só o hash vai ao banco, **rotação a cada uso**; reuso de um token antigo revoga a família inteira; amarrado ao cliente; vida máxima absoluta da família | `full_flow...`, `refresh_token_is_bound_to_its_client`, `expired_refresh_family_is_rejected` |
| **Open redirect / phishing de redirect** | `redirect_uri` com comparação **exata**; cliente ou URI inválidos nunca redirecionam | `invalid_client_or_redirect_never_redirects` |
| **CSRF / clickjacking / XSS no login** | Cookie CSRF `HttpOnly` + `SameSite=Strict` em tempo constante; `frame-ancestors 'none'`; CSP sem `unsafe-inline`; tudo escapado | `login_requires_csrf`, `login_page_is_hardened_and_escapes_input` |
| **Força bruta / credential stuffing** | Argon2id (64 MiB) com *pepper*; bloqueio progressivo **por conta + origem** (IPv4 ou prefixo /64 do IPv6; 5 falhas → 60 s, dobrando até 1 h); limite por IP; resposta idêntica para usuário existente, inexistente e bloqueado (hash falso roda nos três casos) | `wrong_password_is_generic_and_locks_account...`, `blocked_and_wrong_password_look_the_same`, `ipv6_attacker_cannot_dodge...`, `rate_limit_on_auth_endpoints...` |
| **Trancar a conta de outra pessoa** | O bloqueio de uma origem não afeta as outras. Bloqueio da conta inteira só com 50 falhas em 24 h (ataque distribuído) e **não vale para origens onde o usuário já entrou** nos últimos 90 dias; gera alerta | `attacker_on_one_ip_cannot_lock_the_user_out_from_another`, `distributed_attack_locks_the_account_for_unknown_origins_only` |
| **Senha vazada** | Segundo fator TOTP opcional por usuário (`mfa-enable`); segredo cifrado (ChaCha20-Poly1305, chave derivada do *pepper*, amarrado ao id do usuário); cada código vale uma vez; o código só é conferido com a senha certa | `mfa_requires_a_valid_code_once`, `mfa_code_never_rescues_a_wrong_password`, `mfa::tests` (vetores da RFC 6238) |
| **Token roubado em uso** | Reuso de refresh token ou de authorization code derruba **também os access tokens** na hora (`tokens_valid_after`); usuário desativado perde o acesso na hora; `revoke-sessions` para o operador | `refresh_token_theft_kills_the_access_tokens_immediately`, `revoke_all_sessions_and_deactivation_take_effect_at_once` |
| **Vazamento do `JWT_SECRET`** | `kid` em todo token; troca do segredo sem derrubar ninguém: o novo assina, o antigo (`JWT_SECRET_PREVIOUS`) só valida até os tokens expirarem; token sem `kid` ou com `kid` desconhecido é recusado | `jwt_secret_rotation_accepts_old_tokens_only_while_configured`, `tokens::tests` |
| **Flood de requisições** | Limite por IP (geral e mais estrito em `/oauth/*`), limite por usuário no OCR, teto de requisições em andamento com *load shedding* (503) | `general_rate_limit...`, `forged_forwarded_for...`, `load_shedding_returns_503...` (`tests/dos.rs`) |
| **Slowloris / conexões ociosas** | Timeout de leitura de cabeçalhos (10 s), timeout de corpo ocioso (15 s), timeout total por rota, tetos de conexões globais e **por IP**, limites de HTTP/2 e *rapid reset* | `slowloris_connection_is_dropped`, `per_ip_connection_cap`, `huge_headers_are_rejected`, `http2_*` (`tests/dos.rs`) |
| **Corpos gigantes** | Limite por rota: 1–16 KB nas rotas JSON/forms; upload do OCR limitado e interrompido ao estourar | `oversized_bodies_are_rejected` |
| **Upload malicioso** | Lista de extensões; conferência de assinatura (*magic bytes*); limite de páginas; **bomba de pixels** (cabeçalho lido sem decodificar); **bomba de ZIP** (diretório central, sem descompactar); apenas um campo `file`; HTML/XML/SVG/áudio recusados; nome sanitizado, em diretório temporário exclusivo e apagado ao fim | `tests/ocr_upload.rs` (12 testes) |
| **SSRF pelo conversor** | O motor nunca resolve imagens remotas/locais referenciadas por documentos (`ImageSources::None`) e formatos que as referenciam não são aceitos | `disallowed_extensions_are_rejected` |
| **Parser comprometido (RCE via upload)** | Inspeção e conversão rodam num **processo worker por documento**: ambiente zerado (nenhum segredo/URL), Landlock próprio (grava só no diretório do próprio job, não vê uploads alheios, **nenhuma porta TCP**), seccomp (sem `execve`, sem socket de rede, sem ptrace/namespaces/eBPF/montagem), `rlimit` de memória/CPU/arquivos, sem core dump. Cache de grafos ONNX confiável é só leitura (cada job tem uma cópia privada descartável): nada que um job grave chega ao próximo | `worker_profile_is_locked_down` (+ controle `without_sandbox...`), `worker_gets_an_empty_environment_without_secrets`, `crashed_worker_gives_generic_error_alert...`, `garbage_or_oversized_reply_is_rejected` |
| **Processo da API** | Landlock + seccomp (sem ptrace, `process_vm_*`, namespaces, eBPF, montagem, módulos) + `PR_SET_DUMPABLE=0` (memória e `environ` inacessíveis a outros processos do mesmo usuário, sem core dump); segredos removidos do ambiente após a leitura | `api_profile_keeps_what_the_api_needs_and_blocks_the_rest` |
| **Esgotamento por OCR** | 1 OCR por vez por usuário, N globais, espera máxima de 2 s por vaga (senão 503 + `Retry-After`), prazo por documento; ao estourar o prazo (ou o cliente desistir) o worker leva **SIGKILL** e só então a vaga volta | `one_ocr_at_a_time_per_user_and_global_capacity`, `hung_worker_is_killed_at_the_deadline_and_slots_are_freed`, `real_worker_out_of_memory_is_contained` |
| **Abuso do chunking** | Corpo lido só depois do limite de taxa e da vaga do usuário (1 por vez), com teto e prazo; texto, documentos, metadados (copiados para cada chunk) e chunks gerados limitados; estimativa de chunks antes do worker; sobreposição até metade do tamanho; o cliente não passa opções internas (sem SSRF por URL de LLM, sem ler arquivo por `model_name`); divisão num worker isolado com prazo | `tests/chunk.rs` (`invalid_requests_are_rejected_without_details`, `limits_on_text_size_documents_and_chunks`, `hostile_text_only_hurts_the_worker`, `crashed_chunk_worker_gives_a_generic_error_and_an_alert`) |
| **Vazamento de informação** | Mensagens fixas, sem caminhos/SQL/stack; `invalid_grant` único; segredos nunca logados; a query string não entra nos logs | `corrupt_office_file_gives_generic_error`, `ocr_happy_path_returns_text_and_hides_server_paths` |
| **Ataque sem ninguém ver** | Todo evento vai para o `audit_log` **e** para o log (`target=security`, JSON com `LOG_FORMAT=json`). Reuso de token, bloqueio de conta inteira, worker morto por sinal e muitas falhas de login do mesmo IP (`SECURITY_ALERT_LOGIN_FAILURES` em 5 min) geram `target=security_alert` em nível ERROR — pendure o alerta do coletor de logs nele | `repeated_failures_from_one_ip_raise_a_security_alert`, `security::tests` |
| **CVE em dependência** | `cargo-deny` (RustSec, crates retiradas, licenças, origem) no CI a cada push **e diariamente**; Dependabot para cargo, Docker e Actions | `.github/workflows/ci.yml`, `deny.toml` |
| **Banco em texto puro** | Fora de loopback/socket, `DATABASE_URL` sem `sslmode=require/verify-*` impede a subida (opt-out explícito: `DATABASE_ALLOW_PLAINTEXT=true`, usado só na rede interna isolada do compose) | `config::tests::remote_database_requires_tls` |
| **Cabeçalhos** | HSTS (com `https`), `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, CSP padrão restrita, `Cache-Control: no-store`, CORS fechado por padrão, `X-Request-Id` sempre gerado no servidor | `token_responses_are_not_cacheable...`, `cors_is_closed_by_default` |

Teste de mutação: desligar o limite de ZIP, a checagem de pixels, o CSRF e a detecção de reuso do refresh
faz o teste correspondente **falhar** (os testes de upload conferem também o motivo da rejeição).

### Rodar os testes

```bash
export TEST_ADMIN_DATABASE_URL="postgres://api_ocr:<senha de deploy/.env>@127.0.0.1:55432/api_ocr"
export TEST_REDIS_URL="redis://:<senha de deploy/.env>@127.0.0.1:56379/"
cargo test --workspace      # cada teste da API cria e apaga o próprio banco
cargo deny check            # dependências (cargo install cargo-deny --locked)
```

## Operação de segurança

```bash
api_ocr mfa-enable --email voce@exemplo.com      # imprime a URI otpauth:// (QR code) e a chave manual
api_ocr mfa-disable --email voce@exemplo.com
api_ocr revoke-sessions --email voce@exemplo.com # derruba access E refresh tokens na hora
api_ocr unlock-user --email voce@exemplo.com     # remove os bloqueios de login
```

**Trocar o `JWT_SECRET` (vazamento ou rotina):** gere o novo, mova o atual para `JWT_SECRET_PREVIOUS`,
reinicie; depois de `ACCESS_TOKEN_TTL_SECS` (10 min), remova `JWT_SECRET_PREVIOUS` e reinicie de novo.
Se o segredo **vazou**, pule a fase de transição (deixe `JWT_SECRET_PREVIOUS` vazio): todos os access
tokens caem e os clientes renovam com o refresh token, que não depende do segredo.

**Worker de OCR:** cada documento roda num processo `api_ocr ocr-worker` (interno). Na subida, um
warm-up converte um documento sintético embutido e grava o cache de grafos ONNX (~4 s em release);
depois disso cada job custa ~0,25 s a mais que no desenho antigo (medido: 2,55 s contra ~2,3 s por
documento). **Memória:** um vigia na API mede, a cada 10 ms, a memória não recuperável do
contêiner (cgroup v2) e o RSS de cada worker; a 85% do limite, o maior worker leva SIGKILL (a API
responde 422 e registra `ocr_worker_memory_killed`). `OCR_WORKER_MAX_RSS_MB` acrescenta um teto por
worker. Não dá para deixar isso com o kernel: no Kubernetes (`memory.oom.group=1`) o OOM mata o
contêiner INTEIRO, API junto; fora de contêiner, o worker é a primeira vítima (`oom_score_adj=1000`).
Também não use `RLIMIT_AS` (`OCR_WORKER_MEMORY_MB`, desligado): o ONNX Runtime reserva ~7 GB de
endereços com qualquer documento (medido) e **trava**, em vez de falhar, quando uma reserva é negada.
GPU testada: OCR sob `SANDBOX=required` usando a VRAM de verdade. Com muitos núcleos e sem limite de
CPU, fixe `DOCLING_RS_PDF_WORKERS` (cada worker de página custa ~0,4 GB em PDFs de 6+ páginas).

## Produção

```bash
cd deploy
cp .env.prod.example .env.prod          # preencha (segredos: `api_ocr gen-secret`)
# coloque fullchain.pem e privkey.pem reais em deploy/certs/  (./gen-selfsigned.sh só para teste)
docker compose -f docker-compose.prod.yml --env-file .env.prod up -d --build
```

Pilha: **nginx** (TLS, limites por IP, só as rotas da API) → **app** (sandbox Landlock) → **Postgres** e
**Redis**. A rede `internal` é `internal: true`: app, banco e Redis **não têm rota para a internet**; só o
proxy publica portas. O app roda como uid 10001, raiz somente leitura, todas as capacidades removidas,
`no-new-privileges`, 6 GB de memória, 4 CPUs.

Firewall e kernel do host (opcional, recomendado): `deploy/host-hardening/nftables.conf` e
`99-api-ocr.sysctl.conf` (instruções nos próprios arquivos).

## O que foi resolvido, como foi provado e o que continua fora do alcance

| Limite | O que foi implementado | Prova |
|---|---|---|
| **DDoS volumétrico** | `nftables` de borda (inundação de conexões novas por IP e conexões simultâneas por IP, pacotes inválidos, SSH) + `sysctl` (SYN cookies etc.) + limites por IP no nginx. | Em dois contêineres, 200 conexões novas seguidas: 65 passaram (a rajada) e 135 foram derrubadas; o IP entrou no conjunto de bloqueio por 10 min. As regras carregam no kernel e são idempotentes. O `sysctl` **não** foi exercitado, só entregue. |
| **Saturação do link (L3/L4)** | **Não é resolvível no host nem na aplicação.** Exige CDN/WAF ou *scrubbing* do provedor. Se usar Cloudflare, ative `set_real_ip_from` + `real_ip_header CF-Connecting-IP` no nginx e **aceite só IPs da Cloudflare** no firewall. | — |
| **TLS** | nginx termina TLS 1.2/1.3 (HSTS, sem *session tickets*, SNI desconhecido derruba o handshake, HTTP→HTTPS com host fixo), repassa o IP real e **sobrescreve** `X-Forwarded-For`. | Fluxo OAuth+OCR completo por HTTPS com certificado autoassinado; cookie `__Host-csrf; Secure`; `X-Forwarded-For` forjado ignorado (o log de auditoria guarda o IP real). A emissão do certificado real (Let's Encrypt etc.) é com você. |
| **Limite de taxa por instância** | Limite **global** em Redis (token bucket atômico em Lua, relógio do próprio Redis). Cai para o limite local se o Redis falhar, com disjuntor (não adiciona latência). Configurado e inalcançável na partida = erro. | 2 réplicas compartilhando o Redis: só a rajada global passa; sem Redis, passam todas. Queda do Redis simulada: a API continua respondendo e limitando localmente. Mutação: desligar o global quebra o teste. |
| **OCR executa código nativo** | (1) **Landlock** no processo: sem `.env`, `~/.ssh` nem código-fonte; escrita só em `/tmp` e caches; TCP só para Postgres/Redis/DNS. (2) **Controle de memória**: recusa novo OCR (503) acima de 85% do cgroup. (3) Contêiner endurecido (acima). | Processo filho tenta ler segredos, gravar fora e conectar em porta não permitida: tudo `DENIED`; sem sandbox, tudo funciona (controle). OCR de PDF real passa sob `SANDBOX=required` em CPU, GPU e **dentro do contêiner**. |
| **Teto de requisições / HTTP2** | Sem teste antes. | Teto: com 2 requisições presas e `MAX_IN_FLIGHT=2`, a seguinte recebe 503 + `Retry-After` e volta ao normal depois. HTTP/2: servidor anuncia 32 streams e **corta uma inundação de *rapid reset*** (GOAWAY) sem cair. |

**Bug real encontrado ao escrever esses testes:** o teto de requisições em andamento *nunca limitava nada*.
O axum aplica cada camada em cada rota, e `concurrency_limit()` cria um semáforo por aplicação. Foi
trocado por `GlobalConcurrencyLimitLayer` (um semáforo só). Quem usa `tower` com axum pode ter o mesmo problema.

### O que ainda NÃO está resolvido (leia)

1. **Ataque volumétrico real** continua precisando de CDN/provedor, como acima.
2. **Escape de kernel.** Landlock, seccomp e `rlimit` dependem do kernel; uma falha nele escapa de tudo
   isto. Para isolamento contra isso: gVisor (`runsc`) ou microVM (Firecracker) no worker.
3. **O seccomp é lista de negação**, não de permissão: fecha os caminhos de escape conhecidos sem
   arriscar quebrar o ONNX Runtime; uma lista de permitidos exata seria mais forte e mais frágil.
4. **A GPU não foi testada dentro do contêiner** (a imagem é CPU). A GPU foi testada sob o sandbox no host.
5. **Limites locais continuam por instância** quando não há Redis; o limite de OCR *em andamento por
   usuário* (1 por vez) também é por instância.
6. Ressalvas que seguem valendo: cadastro público revela e-mails (409, desligado por padrão), refresh
   simultâneo derruba a sessão (e agora também os access tokens do usuário), `JWT_SECRET` simétrico,
   o *pepper* não tem rotação (trocá-lo invalida senhas e segredos TOTP), MFA é opcional e ativado
   pelo operador, a contagem de falhas por IP para alertas é por réplica.
8. **O vigia de memória mede a cada 10 ms:** uma alocação muito rápida pode chegar ao limite do
   contêiner entre duas medições, e aí o OOM do kernel derruba o contêiner inteiro (o Kubernetes o
   reinicia). Deixe folga entre o consumo de pico dos jobs e `limits.memory`.
9. **A GPU não foi testada DENTRO do contêiner/Kubernetes** (só no host, sob o sandbox).
7. As imagens base e os modelos ONNX vêm da internet no *build* (o script de modelos tem hash fixado;
   os modelos em si não têm checksum publicado).
