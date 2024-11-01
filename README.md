# Dashboard de Monitoramento de Processos do NUPAM

Este repositório apresenta uma aplicação em **Streamlit** voltada para o monitoramento e análise de processos do **NUPAM** no ano de 2024. A aplicação oferece uma visão abrangente do desempenho e da gestão de processos, além de realizar análises preditivas com aprendizado de máquina. 

---

## Funcionalidades

### 1. Resumo Geral
- Exibe o total de processos atendidos, a média de dias que cada processo permaneceu no NUPAM, o total de peças produzidas e o número de tipos diferentes de peças.
- Indicadores de qualidade dos dados são calculados, incluindo a contagem de dados faltantes, incorretos e válidos, exatidão percentual e margem de erro.

### 2. Visualizações Interativas
- **Processos por Cidade**: Gráfico de barras horizontal que exibe o total de processos em cada cidade, permitindo uma análise geográfica dos atendimentos.
- **Processos por Mês**: Gráfico de barras que mostra o total de processos por mês, facilitando a compreensão da distribuição temporal.
- **Distribuição da Duração dos Processos**: Boxplot interativo que destaca a distribuição da duração dos processos, com estatísticas de mediana, quartis e outliers.

### 3. Qualidade dos Dados
- **Contagem de Dados**: Total de registros e detalhamento de dados faltantes ou incorretos, principalmente na coluna de duração dos processos.
- **Indicadores de Exatidão**: Percentual de dados válidos e margem de erro calculada para refletir a precisão dos dados analisados.

### 4. Resumo Estatístico
- Exibe estatísticas como contagem, média, mediana, desvio padrão, valores mínimos e máximos, além dos percentis 25%, 50% e 75% da coluna de duração dos processos.

---

## Análises e Modelos de IA

A aplicação inclui diversas análises e previsões utilizando aprendizado de máquina, possibilitando uma análise mais profunda dos dados:

### 1. Regressão Linear
- **Objetivo**: Identificar tendências na duração dos processos ao longo do tempo.
- **Modelo**: Utiliza regressão linear para encontrar uma relação entre o índice dos processos e sua duração, exibindo uma linha de tendência.
  
### 2. Clustering com KMeans
- **Objetivo**: Agrupar processos com base em sua duração.
- **Modelo**: KMeans com 3 clusters, onde cada grupo representa um conjunto de processos com características similares de duração. A visualização dos clusters facilita a identificação de padrões e anomalias nos tempos de processamento.

### 3. Previsão Temporal com ARIMA
- **Objetivo**: Prever a duração dos processos com base em dados históricos.
- **Modelo**: ARIMA, configurado para previsões de curto prazo. Utiliza dados temporais da duração dos processos para antecipar a duração média nos próximos cinco dias.

### 4. Redes Neurais Artificiais
- **Objetivo**: Prever a duração dos processos com um modelo de redes neurais.
- **Modelo**: Rede neural sequencial com camadas densas, projetada para aprender padrões complexos na duração dos processos. A rede é treinada para prever a duração normalizada dos processos, ajudando a entender o comportamento geral dos dados e possíveis mudanças.

---

## Explicações dos Gráficos e Tabelas

- **Boxplot de Duração dos Processos**: Apresenta a distribuição da duração dos processos, incluindo mediana, quartis e possíveis outliers.
- **Gráfico de Processos por Cidade**: Distribuição geográfica dos processos, destacando as cidades com maior volume.
- **Gráfico de Processos por Mês**: Análise temporal que indica o fluxo de processos ao longo do ano.
  
---

Este dashboard fornece uma ferramenta intuitiva e detalhada para gestores e analistas acompanharem o andamento dos procedimentos e tomarem decisões baseadas em dados, contando ainda com previsões que auxiliam no planejamento estratégico. 
