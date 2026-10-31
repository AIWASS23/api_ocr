# Acessibilidade

O OCR API deve permitir que pessoas com diferentes formas de leitura, navegação
e interação consigam usar sua documentação, autenticar-se e integrar a API.
Este documento cobre a página de login OAuth2, o Swagger UI opcional, as
respostas e exemplos da API e a participação no repositório.

## Situação atual e prioridades

A maior parte do projeto é uma API HTTP com respostas JSON. As interfaces web
mantidas aqui são o login em `/oauth/authorize` e a documentação em `/docs`,
disponível apenas quando `ENABLE_DOCS=true`. A especificação `/openapi.json`
também depende dessa configuração.

O login possui idioma `pt-BR`, rótulos associados aos campos de e-mail, senha
e código de verificação, atributos de preenchimento automático e mensagens
de erro em texto, anunciadas como alerta (`role="alert"`). Esses recursos ajudam
a interação, mas não substituem uma avaliação com usuários e tecnologias assistivas.

As respostas de erro da API trazem um código estável em `error` e uma mensagem
em português em `error_description`. Clientes podem usar o código para mostrar
mensagens próprias, no idioma e no formato adequados a quem usa a aplicação.

As prioridades para alterações nessas interfaces são:

- Permitir navegação e envio de formulários pelo teclado, com foco visível.
- Manter nomes claros para campos e botões e mensagens de erro compreensíveis.
- Preservar leitura com ampliação de texto, contraste adequado e uso em telas pequenas.
- Oferecer instruções em texto e exemplos copiáveis, sem depender só de imagens,
  cores ou gravações para transmitir informação.
- Documentar os campos e erros da API para facilitar clientes acessíveis.

Não há avaliação formal de conformidade WCAG registrada no repositório nem
matriz validada de navegadores, leitores de tela ou dispositivos. O CI atual
valida contratos e comportamento da API, mas não faz auditoria de acessibilidade
das interfaces web.

## Orientações para contribuições

Ao alterar login ou documentação web, verifique a tarefa afetada usando apenas
teclado e com zoom ampliado. Confira rótulos, ordem de foco, mensagens de erro
e leitura em telas menores; quando possível, verifique também com leitor de tela.
Registre no PR o fluxo, o navegador, o sistema e as tecnologias assistivas usados,
os resultados e o que não foi testado; o [template de PR](.github/pull_request_template.md)
tem um item para essa verificação.

Em Markdown, use títulos em ordem lógica, links descritivos, texto alternativo
para imagens informativas e blocos de código com comandos copiáveis. Capturas
de tela devem complementar a explicação em texto. Siga também o
[guia de contribuição](CONTRIBUTING.md).

## Relatar uma barreira

Abra uma [issue de acessibilidade](https://github.com/AIWASS23/api_ocr/issues/new?template=accessibility.md);
o formulário pede só o necessário e nenhum campo é obrigatório.
Se preferir um canal privado, escreva para
[marceloaiwass666@gmail.com](mailto:marceloaiwass666@gmail.com) com o assunto
`[api_ocr][acessibilidade]`.

Descreva a tarefa que tentou realizar, a página ou trecho da documentação,
o resultado esperado, a barreira encontrada e uma alternativa, se houver.
Navegador, sistema e tecnologia assistiva ajudam a reprodução, quando conhecidos.
Imagens e gravações são opcionais; não é necessário informar deficiência ou
qualquer dado de saúde. Remova dados pessoais, documentos privados e credenciais.

Os mantenedores avaliam o impacto na tarefa e priorizam barreiras que impedem
autenticação, acesso à informação ou contribuição. O acompanhamento ocorre na
issue ou pelo e-mail usado no relato, com possíveis alternativas e pedidos de
verificação da correção. Não há prazo fixo de resposta ou resolução.

## Limitações e manutenção

Não há resultados documentados de testes sistemáticos com leitores de tela.
O Swagger UI depende de um componente de terceiros e também precisa de
verificação manual quando atualizado. Como alternativa à navegação nessa UI,
a API pode ser consultada por um cliente HTTP e pelos exemplos no
[README da API](api/README.md), respeitando a autenticação exigida.

A extração por OCR não garante ordem de leitura correta, descrições de imagens
ou documentos finais acessíveis. Revise o conteúdo extraído antes de usá-lo em
interfaces ou materiais destinados a pessoas usuárias.

Os mantenedores do repositório são responsáveis pela triagem e por revisar este
documento quando mudarem as interfaces, os canais de contato ou a cobertura de
testes. Melhorias a esta política podem ser propostas por issue ou PR.

