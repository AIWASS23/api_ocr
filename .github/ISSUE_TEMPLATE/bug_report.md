---
name: Relatar bug
about: Comportamento incorreto da API, da extração, do chunking ou do deploy
title: "[bug] "
labels: bug
assignees: ''

---

<!--
Vulnerabilidade de segurança? Não use este formulário: siga o SECURITY.md e relate em privado.
Não cole tokens, cookies, senhas, arquivos .env, URLs com credenciais, dados pessoais nem
documentos reais. Use arquivos sintéticos ou anonimizados.
-->

## O que aconteceu

## O que era esperado

## Como reproduzir

1.
2.
3.

<!--
Inclua a requisição sem o cabeçalho Authorization e sem cookies, e a resposta (status e corpo).
Para extração: tipo, tamanho e número de páginas do arquivo. Para chunking: estratégia,
chunk_size, chunk_overlap e um texto curto que reproduza o problema.
-->

```text
```

## Componente

- [ ] API HTTP (rotas, validação, limites, OpenAPI)
- [ ] OAuth2, login, sessões ou MFA
- [ ] Extração de documentos (`/v1/data_preparation`, `src/data_preparation`)
- [ ] Chunking (`/v1/chunk`, `src/chunking`)
- [ ] Worker isolado (Landlock, seccomp, prazo, limite de memória)
- [ ] Deploy (Docker, Compose, nginx, Kubernetes)
- [ ] Documentação
- [ ] Outro:

## Ambiente

- Commit, tag ou digest da imagem:
- Execução (local com `cargo`, Docker Compose, Kubernetes, outro):
- Sistema e kernel (`uname -r`):
- CPU ou GPU (feature `cuda`):
- `SANDBOX` (`off`, `auto` ou `required`):

## Logs

<!-- Trecho do log da API; o request_id (cabeçalho X-Request-Id) ajuda a achar a requisição.
Remova tokens, e-mails e conteúdo de documentos. -->

```text
```

## Contexto adicional
