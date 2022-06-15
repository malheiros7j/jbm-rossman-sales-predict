# Rossman Drugstore Sales Prediction

<!-- COLOCAR IMAGEM ROSSMAN -->
# Introdução 
Esse é um projeto end-to-end de Data Science com modelo de regressão adaptada para séries temporais. No qual criamos 4 tipos de modelos para predizer o valor da vendas das lojas nas próximas 6 semanas. As previsões podem ser acessadas pelo usuário por meio de um BOT no aplicativo do Telegram.

Esse repósitório contém a solução para a resolução de uma problema do Kaggle: https://www.kaggle.com/c/rossmann-store-sales 

Esse projeto faz parte da "Comunidade DS" que é um ambiente de estudo que promove o aprendizado, execução, e discussão de projetos de Data Science.

### Plano de Desenvolvimento do Projeto de Data Science
Esse projeto foi desenvolvido seguindo o método CRISP-DS(Cross-Industry Standard Process - Data Science). Essa é uma metodologia capaz de transformar os dados da empresa em conhecimento e informações que auxiliam na toma de decisão. A metodologia CRISP-DM define o ciclo de vida do projeto, dividindo-as nas seguintes etapas:
* Entendimento do Problema de Negócio 
* Coleção dos Dados
* Limpeza de Dados
* Análise Exploratória dos Dados
* Preparação dos Dados
* Modelos de Machine Learning e Fine-Tuning.
* Avaliação dos Resultados do Modelo e Tradução para Negócio.
* Modelo em Produção


<!-- ***** COLOCAR IMAGEM ****** -->

### Planejamento
* [1. Descrição e Problema de Negócio](#1-descrição-e-problema-de-negócio)
* [2. Base de Dados e Premissas de Negócio](#2-base-de-dados-e-premissas-de-negócio)
* [3. Estratégia de Solução](#3-estratégia-de-solução)
* [4. Exploration Data Analysis](#4-exploration-data-analysis)
* [5. Seleção do Modelo de Machine Learning](#5-seleção-do-modelo-de-machine-learning)
* [6. Perfomance do Modelo](#6-perfomance-do-modelo)
* [7. Resultados de Negócio](#7-resultados-de-negócio)
* [8. Conclusão](#8-conclusão)
* [9. Próximos Passos](#9-próximos-passos)


# 1. Descrição e Problema de Negócio

### 1.1 Descrição
**Rossman Sales Drugstore** é uma empresa que opera mais de 3000 drogarias em 7 países europeus. Atualmente os gerentes da loja Rossman tem a tarefa de prever suas vendas diárias com até seis semanas de antecedência. As vendas das lojas são influenciadas por muitos fatores, incluindo promoções, competição, feriados escolares e estaduais, sazonalidade e localidade. Atualmente essa previsão é feita por meio de uma simples média das vendas de cada loja. 

### 1.2 Problema de Negócio
Foi feita uma reunião entre o CEO e os sócios da Rossman e foi definido que a empresa irá investir nas reformas das lojas da rede Rossman. Para que essa reforma seja possível será necessário prever o valor de vendas de cada loja de maneira mais assertiva, para que assim o CEO tenho uma melhor noção do quanto investir em cada loja. 

Dito isso,a empresa decidiu contratar um Cientista de Dados para realizar as seguintes tarefas:

**- Realizar a previsão das vendas de cada uma das lojas nas pŕoxima seis semanas.**

**- Fornecer ao CEO uma forma de consulta rápida dessas previsões por meio do celular.**


# 2. Base de Dados e Premissas de Negócio
## 2.1 Base de Dados
O conjunto de dados total possui as informações referentes 1115 lojas e possuem os seguintes atributos:
* Id - um identificador que representa uma dupla (Store, Date) dentro do conjunto de teste.
* Store - identificador único de cada loja
* Sales - O valor das vendas dado o dia.
* Customers - número de clientes dado o dia.
* Open - indicador que informa se a loja está aberta: 0 = fechado, 1 = aberta.
* StateHoliday - Indica um feriado estatal. Geralmente todas lojas, com algumas exceções, fecham em feŕiados estaduais. Note que todas escolas estão fechadas nos feriados e finais de semana. a = Feriado  b = Feriado da Páscoa , c = Feriado de Natal Christmas, 0 = Nenhum
* SchoolHoliday - indica se (Store, Date) foi afetada pelo fechamento da escolas públicas.
* StoreType - Diferencia 4 tipos diferentes de loja: a, b, c, d
* Assortment - Descreve o tipo de nível de estoque: a = básico, b = extra, c = extended
* CompetitionDistance - distância em metros do competidor mais próximo.
* CompetitionOpenSince[Month/Year] - Dá o ano e mês em que o concorrente mais próximo foi aberto.
* Promo - indica se a loja está com alguma promoção em vigẽncia naquele dia.
* Promo2 - Promo2 é uma promocação contínua e consecutiva para algumas lojas: 0 = loja não está participando, 1 = a loja está participando.
* Promo2Since[Year/Week] - descreve o ano e a semana que a loja começou a participar da Promo2.
* PromoInterval - descreve os intervalos consecutivos do início da promo2, nomeando os mês que a promoção é iniciada novamente. Ex:("fev,maio,agosto") significa que cada rodada começa em fav, maio, agosto para qualquer ano dado daquela loja.
## 2.2 Premissas de Negócio
Para realizar esse projeto as seguintes premissas de negócio foram adotadas:
* Os dados de costumers foram descartados, visto que para utilizar esse atributo teríamos que calcular uma previsão de número de clientes que pode-se tornar um projeto a parte complementar a este.
* Os dias que as lojas encontram-se fechadas foram descartadas.
* Só foram considerados as entradas que obtiveram o valor de venda ("SALES") maior que 0.
* Para lojas que não tinham informação de Competition Distance foi adotada um valor arbitrário alto para efeitos de comparação.
# 3. Estratégia de Solução
A estratégia de solução foi a seguinte:
### Passo 01. Descrição dos Dados
Nesse passo foi verificado alguns aspectos do conjunto de dados, como: nome de colunas, dimensões, tipos de dados, checagem e preenchemento de dados faltantes (NA), análise descritiva dos dados e quais suas variáveis categórias.
### Passo 02. Featuring Engineering
Na featuring engineering foi derivado novos atribudos(colunas) baseados nas variáveis originais, possibilitando uma melhor descrição do fenômeno daquela variável.

### Passo 03. Filtragem de Variáveis
O conjunto de dados foi filtrado por linhas para que levassemos em consideração apenas as lojas que estão aberta e que realizaram vendas ( open != 0 e sales > 0) e por coluna foi feita um drop das variáveis que não agregam valor de conhecimento ou foram derivados para outras variáveis.
### Passo 04. Análise Exploratória dos Dados (EDA)
Exploração dos Dados com objetivo de encontrar Insights para o melhor entendimento do Negócio. 
Foram feitas também análises univariadas, bivariadas e multivariadas, obtendo algumas propriedades estatísticas que as descrevem, e mais importante  a correleção entre as variáveis.
### Passo 05. Preparação dos Dados
Sessão que trata da preparação dos dados para que os algoritmos de Machine Learning possam ser aplicados. Foram realizadas alguns tipos de escala e encoding para que as variáveis categóricas se tornasse numéricas.
### Passo 06. Seleção da Variáveis do Algoritmo
A seleção dos atributos foi realizada utilizando o método de seleção de variáveis Boruta. No qual os atributos mais significativos foram selecionados para que a perfomance do modelo fosse maximizada.
### Passo 07. Modelo de Machine Learning
Realização do treinamento dos modelos de Machine Learning . O modelo que apresentou a melhor perfomance diante a base de dados com cross-validation aplicada seguiu adiante para a hiper parametrização das variáveis daquele modelo, visando otimizar a generalização do modelo.
### Passo 08. Hypervfhnjtfdjjjjjjjjjjjj  Parameter Fine Tuning
Foi encontrado os melhores parâmetros que maximizavam o aprendizado do modelo. Esses parâmetro foram definidos com base no método de RandomSearch.
### Passo 09. Conversão do Desempenho do Modelo em Valor de Negócio
Nesse passo o desempenho do modelo foi analisado diante uma perspectiva de negócio,e traduzido para valores de negócio.
### Passo 10. Deploy do Modelo em Produção 
Publicação do modelo em um ambiente de produção em nuvem (Heroku) para que fosse possível o acesso de pessoas ou serviços para consulta dos resultados e com isso melhorar a decisão de negócio da empresa.

### Passo 11. Telegram Bot
Criação de um bot no Aplicativo de mensagens do Telegram. Cuja consulta das previsões podem ser feitas de qualquer lugar a qualquer momento apenas utilizando uma conexão com a internet e o aplicativo no smartphone.

# 4. Exploration Data Analysis 
Analise exploratoria de Dados | Insights | Analises Univarias e Multivariadas | Hipóteses 
## 4.1 Análise Univariada
* Variáveis Numéricas: o histograma abaixo mostra como está organizada a distribuição das variáveis númericas do nosso conjunto de dados.

## 4.2 Análise Bivariada
H2. Lojas com competidores mais proximos deveriam vender menos.
H3. Lojas com competidores à mais tempo deveriam vender mais.
H7. Lojas abertas durante o feriado de Natal deveriam vender mais
H8. Lojas deveriam vender mais ao longo dos anos
H9. Lojas deveriam vender mais no segundo semestre do ano
H10. Lojas deveriam vender mais depois do dia 10 de cada mês
H11. Lojas deveriam vender menos aos finais de semana
,lkl

## 4.3 Análise Multivariada

## 

# 5. Seleção do Modelo de Machine Learning 
Tipos de Modelos Treinados

# 6. Perfomance do Modelo
Perfomance do Modelo de Machine Learning treinado
# 7. Resultados de Negócio
Avaliação do Resultado do modelo como negócio 
# 8. Conclusão
Conclusoes e trabalhos Futuros
# 9. Próximos Passos












