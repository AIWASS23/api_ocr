# Atualizações de dependências

O `dependabot.yml` verifica atualizações todos os dias às 04:00 UTC (01:00 em
Fortaleza) e abre PRs para Cargo, Docker, Docker Compose e GitHub Actions.
Cargo inclui dependências diretas e transitivas de todo o workspace. Atualizações
minor/patch são agrupadas por ecossistema; majors ficam em PRs individuais.
Correções de segurança de Cargo têm um grupo separado, sem restrição a minor/patch.

Depois de integrar a configuração à branch padrão, habilite no GitHub, em
**Settings → Code security**, o dependency graph, Dependabot alerts e Dependabot
security updates. O YAML configura os PRs, mas não habilita essas opções do
repositório. PRs de segurança dependem de um alerta conhecido pelo GitHub e de uma
atualização resolvível; eles não aguardam o cron. Referência:
[configuração de security updates](https://docs.github.com/en/code-security/dependabot/dependabot-security-updates/configuring-dependabot-security-updates).

O workflow `ci.yml` já executa `cargo-deny` e testes em `pull_request`. Ele também
audita diariamente a branch padrão. O Dependabot não interpreta os logs do CI nem
faz merge automático; uma falha permanece até a correção ser integrada.

## Cobertura de infraestrutura

- Docker: bases do `deploy/Dockerfile` e imagens nos manifests Kubernetes.
- Compose: nginx, Postgres e Redis nos dois arquivos de `deploy/`.
- GitHub Actions: referências `uses:` dos workflows.

Debian e nginx usam tag com digest para permitir PRs quando o conteúdo de
`trixie-slim` ou `stable-alpine` mudar. Tags numéricas parciais como `17-alpine` e
`7-alpine` permitem propostas de novas versões, mas não registram reconstruções da
mesma tag sem um digest. A imagem `api-ocr` segue o processo de release do projeto.
O Dependabot não atualiza o cluster Kubernetes, configurações nginx, pacotes
instalados por `apt`, imagens de serviços do workflow nem comandos Helm escritos
no README. Suporte dos ecossistemas:
[documentação do GitHub](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories).

## Vulnerabilidades do Cargo.lock

O lockfile foi atualizado para `ctutils 0.4.7`, `rustls-webpki 0.103.16`,
`webpki-root-certs 1.0.11` e `webpki-roots 1.0.11`. A versão `zip 8.6.0` foi
removida: Docling e seus componentes estão alinhados em `2.53.1`, com
`calamine 0.37.0` e `utoipa-swagger-ui 10.0.2`, que usam ZIP 9. O ONNX Runtime
foi alinhado em `2.0.0-rc.15`, compatível com Docling e fastembed.
`pulldown-cmark` também foi atualizado para `0.13.6` para corrigir o alerta
RUSTSEC-2026-0414 encontrado pela auditoria atualizada.

Atualizar apenas o `zip` direto não basta quando outras bibliotecas exigem uma
versão vulnerável. Ao atualizar Docling, revise também a tag e o hash do script
de download dos modelos no Dockerfile. Não adicione exceções em `deny.toml` para
esconder o alerta.

Cada PR de correção deve passar pela auditoria e pelos testes existentes antes do
merge. Uma atualização de imagem também exige validação de build/deploy: os testes
Rust atuais não validam a infraestrutura.
