# Como contribuir com o OCR API

O `api_ocr` é um workspace Rust para extração de documentos com OCR e divisão de
texto em chunks para RAG. A API HTTP usa axum/hyper, autenticação OAuth2 com PKCE,
PostgreSQL e, opcionalmente em execução, Redis para limites de taxa compartilhados.

Contribuições de código, testes, documentação e acessibilidade são bem-vindas.
Ao participar, siga o [Código de Conduta](CODE_OF_CONDUCT.md). Relate possíveis
vulnerabilidades pelo canal privado da [política de segurança](SECURITY.md).

## Antes de começar

- Consulte as issues e os PRs existentes para evitar trabalho duplicado.
- Use os [templates de issue](https://github.com/AIWASS23/api_ocr/issues/new/choose)
  para relatar bugs, propor funcionalidades ou descrever barreiras de acessibilidade.
  Issues em branco estão desativadas; vulnerabilidades e problemas de conduta têm
  canais privados, indicados na mesma página.
- Para mudanças grandes no contrato HTTP, autenticação, processamento ou deploy,
  descreva primeiro o problema, a proposta e o impacto na compatibilidade em uma issue.
- Use documentos sintéticos ou anonimizados nas reproduções. Não publique tokens,
  senhas, arquivos `.env`, dados pessoais ou documentos de clientes.

Issues e PRs podem ser escritos em português ou inglês. Preserve os nomes dos
campos da API, comandos e variáveis de ambiente ao traduzir explicações.

## Organização do repositório

| Caminho | Responsabilidade |
| --- | --- |
| [`api/`](api/README.md) | Servidor HTTP, OAuth2, usuários, limites, workers e OpenAPI |
| [`api/tests/`](api/tests/) | Testes de integração, isolamento e resistência a entradas maliciosas |
| [`api/migrations/`](api/migrations/) | Migrações PostgreSQL via SQLx |
| [`src/data_preparation/`](src/data_preparation/) | Conversão de documentos e OCR com Docling |
| [`src/chunking/`](src/chunking/README.md) | Estratégias de chunking, modelos e testes de paridade |
| [`deploy/`](deploy/) | Docker, Compose, nginx e configurações do host |
| [`deploy/k8s/`](deploy/k8s/README.md) | Manifests e instruções de Kubernetes |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Build, testes e auditoria de dependências |

## Preparar o ambiente

Execute os comandos a partir da raiz do repositório, em Linux. A API usa recursos
do Linux, como Landlock, seccomp e `/proc`; a cobertura de isolamento depende do
suporte do kernel. O CI roda em Ubuntu 24.04 com Rust stable. Os crates usam a
edição 2024; não há uma versão mínima de Rust declarada nos manifests.

Tenha Rust/Cargo, um linker e ferramentas de compilação C/C++, além de Docker
com Compose para os serviços locais. O build usa bibliotecas nativas de
Docling/ONNX; consulte o [Dockerfile](deploy/Dockerfile) para o ambiente de
referência. A imagem de produção é CPU; CUDA é uma feature opcional.

```bash
cargo build --workspace --all-targets --locked
```

Crie `deploy/.env` com `OCR_DB_PASSWORD` e `OCR_REDIS_PASSWORD`, uma senha local
distinta para cada serviço. O comando `./target/debug/api_ocr gen-secret` gera um
valor aleatório. Use valores compatíveis com URLs ou codifique os caracteres
reservados ao montar as URLs de conexão. Esse arquivo é ignorado pelo Git.

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --wait
```

O Compose publica PostgreSQL em `127.0.0.1:55432` e Redis em
`127.0.0.1:56379`. Para executar a API, siga o [guia da API](api/README.md) e
preencha uma cópia de [`api/.env.example`](api/.env.example) em `api/.env`,
ajustando a URL do banco, o `ISSUER` e os segredos. Os comandos de teste abaixo
usam variáveis próprias e não exigem carregar `api/.env`.

OCR de PDFs e imagens precisa dos modelos locais de Docling em `.models`.
O estágio `models` do [Dockerfile](deploy/Dockerfile) registra o script de download,
a versão e a verificação de hash usados pelo projeto. Execute o servidor na raiz
do repositório para que encontre esses modelos. Não versione modelos ou caches.

## Validar a alteração

Use PostgreSQL e Redis exclusivos de desenvolvimento: cada teste de integração
da API cria e apaga seu próprio banco, e o usuário de `TEST_ADMIN_DATABASE_URL`
precisa ter permissão para isso. Os testes de limites compartilhados exigem Redis.

Se `deploy/.env` contiver as atribuições de senha descritas acima, carregue-as
no mesmo terminal dos testes:

```bash
. ./deploy/.env
export TEST_ADMIN_DATABASE_URL="postgres://api_ocr:${OCR_DB_PASSWORD}@127.0.0.1:55432/api_ocr"
export TEST_REDIS_URL="redis://:${OCR_REDIS_PASSWORD}@127.0.0.1:56379/"
```

A paridade da estratégia `nltk` precisa dos dados `punkt_tab`, incluindo português,
em `~/nltk_data/tokenizers/punkt_tab` ou sob um diretório indicado por `NLTK_DATA`.
O [workflow de CI](.github/workflows/ci.yml) contém os comandos de download e
extração. Sem esses dados, parte da paridade é pulada; informe isso no PR.

Estas são as verificações executadas pelo CI:

```bash
cargo build --workspace --all-targets --locked
cargo test --workspace --locked

# Instale uma vez, se necessário:
cargo install cargo-deny --locked
cargo deny check advisories bans licenses sources
```

O CI não roda o clippy, mas o workspace passa sem avisos. Mantenha assim:

```bash
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Durante o desenvolvimento, execute primeiro a suíte relacionada à alteração:

```bash
cargo test -p chunking --locked
cargo test -p data_preparation --locked
cargo test -p api_ocr --test chunk --locked
cargo test -p api_ocr --test oauth_flow --locked
```

Esses comandos não substituem a suíte completa ao alterar código compartilhado.
Features opcionais e testes marcados como ignorados não são todos exercitados
pelo CI padrão. Se alterar modelos ou integrações opcionais, consulte os
[testes de chunking](src/chunking/README.md#testes) e registre as features,
os serviços externos e os modelos usados. Informe também testes pulados por
falta de suporte do kernel. Para mudanças só de documentação, confira links,
comandos e exemplos; não é necessário executar OCR.

## Cuidados com a implementação

- Mantenha o PR focado e acompanhe o estilo dos arquivos alterados. Evite
  reformatações gerais sem relação com a mudança.
- Para corrigir um bug de código, inclua um teste que reproduza a falha e
  comprove a correção. Para novas estratégias, cubra limites, Unicode,
  metadados e paridade quando aplicável.
- Alterações em rotas ou respostas devem atualizar OpenAPI, os testes de
  documentação e o README da API. Diferencie recursos da biblioteca de
  chunking daqueles disponíveis pela API HTTP.
- Preserve validação de uploads, isolamento dos workers, limites de recursos
  e respostas que não revelem detalhes internos. Explique mudanças nessas garantias.
- Para mudanças de schema, adicione uma migração em `api/migrations/`, preservando
  as já aplicadas. Descreva compatibilidade, implantação e recuperação em caso de falha.
- Ao mudar configuração, atualize [`api/.env.example`](api/.env.example), o
  ConfigMap em [`deploy/k8s/base/`](deploy/k8s/base/) e os arquivos do Compose afetados.
  Mudanças de infraestrutura precisam de validação de build/deploy; os testes
  Rust não validam Docker, nginx ou Kubernetes.
- Mantenha `Cargo.lock` junto de mudanças de dependências. Consulte
  [`deny.toml`](deny.toml) e as [notas de dependências](.github/README.md);
  não silencie alertas apenas para fazer o CI passar.
- Para login, Swagger UI ou documentação, siga as orientações de
  [acessibilidade](ACCESSIBILITY.md) e registre a verificação manual pertinente.

## Enviar um pull request

Crie uma branch a partir de `main` (em um fork, se necessário) e abra o PR para
`main`. Preencha o [template de PR](.github/pull_request_template.md) com o problema,
a mudança, a issue relacionada, os comandos executados e seus resultados. Declare verificações não executadas
e o motivo, além de alterações incompatíveis ou passos de implantação.

Use títulos e mensagens de commit que descrevam a intenção. Os mantenedores
podem pedir ajustes antes da integração; não há prazo fixo de revisão.
As contribuições incorporadas ao projeto seguem a [licença MIT](LICENSE).
