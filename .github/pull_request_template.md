<!--
Guia: CONTRIBUTING.md. Vulnerabilidade? Não descreva a exploração aqui; siga o SECURITY.md.
Não inclua tokens, senhas, arquivos .env, dados pessoais nem documentos reais.
-->

## Resumo

<!-- O problema e a mudança, em poucas linhas. -->

Issue relacionada: #

## Tipo de mudança

- [ ] Correção de bug
- [ ] Nova funcionalidade
- [ ] Mudança incompatível (rota, campo, código de erro, configuração ou comportamento)
- [ ] Refatoração sem mudança de comportamento
- [ ] Documentação
- [ ] Dependências, CI ou deploy

## Componentes afetados

- [ ] API HTTP e OpenAPI (`api/`)
- [ ] OAuth2, sessões ou MFA
- [ ] Extração (`src/data_preparation`, `/v1/data_preparation`)
- [ ] Chunking (`src/chunking`, `/v1/chunk`)
- [ ] Worker isolado e sandbox (`worker.rs`, `sandbox.rs`, `hardening.rs`)
- [ ] Banco (`api/migrations/`)
- [ ] Deploy (Dockerfile, Compose, nginx, Kubernetes)

## Como foi verificado

<!-- Marque só o que rodou e cole o resultado (ex.: "test result: ok. 167 passed"). -->

- [ ] `cargo build --workspace --all-targets --locked`
- [ ] `cargo test --workspace --locked` (com `TEST_ADMIN_DATABASE_URL` e `TEST_REDIS_URL`)
- [ ] `cargo deny check advisories bans licenses sources`
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings` (recomendado; o CI não roda)
- [ ] Suíte específica: `cargo test -p <crate> --test <arquivo> --locked`
- [ ] Build da imagem ou deploy (Docker, Compose, k3d), se mudou algo em `deploy/`
- [ ] Verificação manual de acessibilidade, se mudou o login ou o Swagger UI

Não verificado, e por quê:

## Checklist

- [ ] A correção de bug vem com um teste que falhava antes dela.
- [ ] Rota ou resposta alterada: tipos do OpenAPI (`api/src/models.rs`, `api/src/docs.rs`), `api/tests/docs.rs` e `api/README.md` atualizados.
- [ ] Configuração nova ou alterada: `api/.env.example`, `deploy/k8s/base/configmap.yaml` e os arquivos do Compose atualizados.
- [ ] Schema alterado: migração nova em `api/migrations/`, sem editar as já aplicadas.
- [ ] Dependências: `Cargo.lock` incluído e nenhuma exceção nova no `deny.toml` sem justificativa.
- [ ] Garantias de segurança preservadas (validação de entrada, isolamento do worker, limites, erros sem detalhes internos), ou a mudança está explicada abaixo.
- [ ] Nenhum segredo, dado pessoal ou documento real em código, testes, fixtures ou capturas.

## Implantação e compatibilidade

<!-- Variáveis novas, migrações, impacto em clientes, passos de deploy e como reverter. Escreva "nenhum" se não houver. -->
