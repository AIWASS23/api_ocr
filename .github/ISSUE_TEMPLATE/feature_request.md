---
name: Propor funcionalidade
about: Sugira uma melhoria para a API, a extração, o chunking ou o deploy
title: "[feature] "
labels: enhancement
assignees: ''

---

## Problema

<!-- Que necessidade ou limitação motivou o pedido? Ex.: "Preciso dividir contratos por cláusula, mas a estratégia legal só reconhece artigos." -->

## Proposta

<!-- Como deveria funcionar. Para mudanças na API, mostre a requisição e a resposta esperadas. -->

## Alternativas consideradas

## Componente

- [ ] API HTTP (rotas, validação, limites, OpenAPI)
- [ ] OAuth2, login, sessões ou MFA
- [ ] Extração de documentos (`/v1/data_preparation`, `src/data_preparation`)
- [ ] Chunking (`/v1/chunk`, `src/chunking`)
- [ ] Worker isolado (Landlock, seccomp, prazo, limite de memória)
- [ ] Deploy (Docker, Compose, nginx, Kubernetes)
- [ ] Documentação
- [ ] Outro:

## Impacto

- [ ] Muda o contrato HTTP (rotas, campos ou códigos de erro) ou quebra clientes existentes
- [ ] Precisa de modelo novo, serviço externo ou acesso à rede no worker
- [ ] Altera autenticação, escopos OAuth ou limites de recursos
- [ ] Exige migração de banco ou mudança de configuração ou de deploy
- [ ] Afeta o login, o Swagger UI ou a documentação (veja o ACCESSIBILITY.md)

<!-- Descreva o que marcou: compatibilidade, custo de CPU e memória, efeito na segurança. -->

## Contexto adicional
