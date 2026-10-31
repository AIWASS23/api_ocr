# Política de segurança

Esta política cobre o código e as configurações de implantação do `api_ocr`:
API HTTP, OAuth2, sessões, OCR, chunking, workers, dependências e manifests
mantidos neste repositório.

## Versões e manutenção

Os crates do workspace estão na versão `0.1.0`. As correções são direcionadas
à branch `main`; o projeto não mantém linhas antigas com correções retroativas
nem oferece prazo garantido de suporte ou correção. Ao relatar uma falha,
informe o commit, a tag ou o digest da imagem utilizada, pois a versão do crate
sozinha pode não identificar o código em execução.

## Relatar uma vulnerabilidade

Envie o relato para [marceloaiwass666@gmail.com](mailto:marceloaiwass666@gmail.com),
com o assunto `[api_ocr][segurança]` e uma descrição breve. Esse é o contato do
mantenedor já utilizado pelo projeto. Não abra uma issue ou PR público com
detalhes de exploração antes da triagem e da divulgação coordenada.

Inclua, quando possível:

- Componente afetado, versão/commit e ambiente: Linux/kernel, CPU ou GPU,
  execução local, Docker ou Kubernetes.
- Comportamento observado, impacto e condições necessárias, como autenticação,
  permissões, configuração do proxy ou modo de sandbox.
- Passos reproduzíveis e uma prova de conceito mínima com dados sintéticos.
- Requisições, respostas e trechos de logs sem tokens, cookies, senhas,
  URLs com credenciais, dados pessoais ou conteúdo de documentos privados.
- Uma forma de contato e sua preferência sobre crédito público pela descoberta.

Não é necessário ter uma correção pronta ou determinar a gravidade para relatar.
Teste apenas em ambientes seus ou com autorização de seus responsáveis.

## Tratamento dos relatos

O mantenedor avalia a reprodução, o impacto e as versões afetadas, e pode pedir
informações adicionais pelo mesmo canal. Quando a falha é confirmada, a correção
e sua divulgação são coordenadas com quem a relatou. O retorno deve indicar o
resultado da análise, uma correção ou mitigação disponível, ou o motivo de o
relato não ser considerado uma vulnerabilidade do projeto.

Não há equipe de plantão nem SLA de resposta. Os prazos dependem da gravidade,
da complexidade e da disponibilidade dos mantenedores. Se não receber retorno,
reenvie a mensagem no mesmo tópico. Evite publicar detalhes exploráveis enquanto
a coordenação estiver em andamento.

## Controles e limites conhecidos

O [README da API](api/README.md) descreve as defesas implementadas, seus testes
e as limitações operacionais. Entre os pontos relevantes para avaliar um relato:

- OAuth2 usa Authorization Code com PKCE S256; sessões, rotação de refresh
  tokens, JWT e MFA possuem testes específicos em `api/tests/`.
- OCR e chunking processam entradas em workers com limites e isolamento.
  Landlock e seccomp dependem do ambiente e do kernel. `SANDBOX=auto` pode
  aplicar proteção parcial; `SANDBOX=required` exige a aplicação do sandbox
  verificado na inicialização, mas não elimina os limites descritos no README.
- Limites de taxa compartilhados dependem de Redis. Se ele falhar durante a
  execução, o serviço usa limites locais por réplica. Limites da aplicação
  não protegem contra saturação do link de rede.
- Modelos e bibliotecas nativas fazem parte da superfície de ataque. O script
  de download dos modelos tem hash fixado no Dockerfile; os modelos baixados
  não possuem a mesma verificação individual documentada. Os dados `punkt_tab`
  do NLTK, usados pelo chunking, vêm de um commit fixo e são conferidos por SHA-256.
- No chunking, o cliente escolhe só a estratégia, o `chunk_size` e o
  `chunk_overlap`. Modelos, caminhos e URLs são configuração do servidor;
  um relato que consiga influenciá-los a partir da requisição é relevante.
- A configuração de TLS, proxy, rede, segredos e recursos do ambiente também
  afeta a segurança. Consulte os exemplos de [deploy](deploy/) e
  [Kubernetes](deploy/k8s/README.md) antes de expor uma instância.

O [CI](.github/workflows/ci.yml) executa `cargo-deny` e testes, inclusive em uma
rotina diária. Isso ajuda a detectar problemas conhecidos, mas não certifica
a ausência de vulnerabilidades nem atualiza instâncias já implantadas.

## Fora do escopo

- Ataques volumétricos (DDoS) contra uma instância; os limites da aplicação não
  substituem proteção de rede.
- Instâncias operadas por terceiros: relate a quem as administra.
- Vulnerabilidades já públicas em dependências, sem impacto demonstrado no
  projeto: relate ao projeto de origem. O `cargo-deny` do CI acompanha os alertas
  do RustSec.
- Limitações já listadas na seção "O que ainda NÃO está resolvido" do
  [README da API](api/README.md), como escape de kernel ou o seccomp por lista
  de negação, salvo quando o relato demonstra um caminho novo de exploração.

Erros funcionais sem impacto de segurança podem ser relatados pelos
[templates de issue](https://github.com/AIWASS23/api_ocr/issues/new/choose).
Em caso de dúvida sobre exposição de dados ou quebra de isolamento, use o e-mail.
