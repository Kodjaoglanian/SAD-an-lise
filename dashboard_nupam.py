import streamlit as st
import pandas as pd
import requests
from io import StringIO
import plotly.express as px
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import calendar

# Carregar variáveis de ambiente
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
REPO_URL = "https://api.github.com/repos/Kodjaoglanian/Ceippam-Sinova/contents/processos.csv"

# Função para baixar o arquivo CSV do GitHub e remover colunas 'Unnamed'
def download_csv(token, url):
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    response_json = response.json()
    
    if 'download_url' not in response_json:
        raise KeyError("The key 'download_url' was not found in the response.")
    
    csv_content = requests.get(response_json['download_url']).content.decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))

    # Remover colunas 'Unnamed' imediatamente após o carregamento
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

# Baixar e carregar o CSV
df = download_csv(TOKEN, REPO_URL)

# Função para separar os tipos de peças, garantindo que só strings sejam processadas
def separar_pecas(pecas_str):
    if isinstance(pecas_str, str):
        return pecas_str.split(' + ')
    else:
        return [pecas_str] if pd.notna(pecas_str) else []

# Aplicar a função para separar as peças
df['PEÇAS ELABORADAS'] = df['PEÇAS ELABORADAS'].apply(separar_pecas)

# Tenta converter as datas, ignorando erros
df['Data de entrada'] = pd.to_datetime(df['Data de entrada'], format='%d/%m/%Y', errors='coerce')
df['Data remessa'] = pd.to_datetime(df['Data remessa'], format='%d/%m/%Y', errors='coerce')

# Filtra o ano atual
current_year = datetime.datetime.now().year
df['Ano'] = df['Data de entrada'].dt.year

# Total de processos atendidos em 2024
df_2024 = df[df['Ano'] == 2024]
total_processos = df_2024.shape[0]

# Cria a coluna Duração
df_2024['Duração'] = (df_2024['Data remessa'] - df_2024['Data de entrada']).dt.days

# Filtrar colunas que não estão totalmente em branco
valid_columns = df.columns[df.notna().any()]
df_valid = df[valid_columns]

# Total de dados
total_dados = df_valid.shape[0]

# Contagem de dados faltantes (valores NaN em qualquer coluna)
dados_faltantes = df_valid.isnull().sum().sum()

# Contagem de dados incorretos (NaN na coluna 'Duração')
dados_incorretos = df_2024['Duração'].isnull().sum()

# Total de dados válidos: apenas contagem de linhas onde Duração não é NaN
dados_validos = df_2024[df_2024['Duração'].notnull()]
total_validos = dados_validos.shape[0]

# Exatidão percentual
exatidao_percentual = (total_validos / total_dados) * 100 if total_dados > 0 else 0

# Margem de erro (5% dos dados válidos)
margem_erro = 0.05 * total_validos

# Média de dias que o procedimento passou no NUPAM
media_duracao = df_2024['Duração'].mean() if total_processos > 0 else 0

# Processos por cidade (Município)
processos_por_cidade = df_2024['Município'].value_counts().reset_index()
processos_por_cidade.columns = ['Município', 'Total de Processos']

# Listar todas as cidades possíveis (pode ser de um arquivo separado ou uma lista fixa)
todas_cidades = df['Município'].unique()
processos_por_cidade_full = pd.DataFrame({'Município': todas_cidades})
processos_por_cidade_full = processos_por_cidade_full.merge(processos_por_cidade, on='Município', how='left').fillna(0)

# Ordenar os dados de processos por cidade
processos_por_cidade_full['Total de Processos'] = processos_por_cidade_full['Total de Processos'].astype(int)
processos_por_cidade_full = processos_por_cidade_full.sort_values('Total de Processos', ascending=False)

# Processos por mês
processos_por_mes = df_2024['Data de entrada'].dt.month_name().value_counts().reset_index()
processos_por_mes.columns = ['Mês', 'Total de Processos']

# Ordenar processos_por_mes pela ordem dos meses
processos_por_mes['Mês_Num'] = processos_por_mes['Mês'].apply(lambda x: list(calendar.month_name).index(x))
processos_por_mes_sorted = processos_por_mes.sort_values('Mês_Num')

# Análise de peças
pecas = df_2024.explode('PEÇAS ELABORADAS').groupby('PEÇAS ELABORADAS')['QNTD'].sum().reset_index()
pecas = pecas.rename(columns={'PEÇAS ELABORADAS': 'Tipo de Peça', 'QNTD': 'Quantidade'})
total_pecas = pecas['Quantidade'].sum()
total_tipos_pecas = pecas.shape[0]

# Ensure the 'PEÇAS ELABORADAS' column remains as lists and remove incorrect conversion to numeric
# Remove or comment out the following lines:
# df_2024.loc[:, 'PEÇAS ELABORADAS'] = pd.to_numeric(df_2024['PEÇAS ELABORADAS'], errors='coerce').fillna(0)

# Remove duplicate calculations of 'Duração' and datetime conversion if any
# ...existing code...

# Filtra procedimentos não em branco com listas não vazias em 'PEÇAS ELABORADAS'
df_filtered = df_2024[df_2024['PEÇAS ELABORADAS'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

# Filtrar colunas que não estão totalmente em branco, garantindo que 'Duração' seja mantida
valid_columns = df_filtered.columns[df_filtered.notna().any()]
if 'Duração' not in valid_columns:
    valid_columns = valid_columns.tolist() + ['Duração']
df_valid = df_filtered[valid_columns]

# Total de dados (excluindo procedimentos em branco)
total_dados = df_valid.shape[0]

# Contagem de dados faltantes (valores NaN em qualquer coluna) excluindo procedimentos em branco
dados_faltantes = df_valid.isnull().sum().sum()

# Contagem de dados incorretos (NaN na coluna 'Duração') excluindo procedimentos em branco
dados_incorretos = df_valid['Duração'].isnull().sum()

# Total de dados válidos: apenas contagem de linhas onde Duração não é NaN excluindo procedimentos em branco
dados_validos = df_valid[df_valid['Duração'].notnull()]
total_validos = dados_validos.shape[0]

# Exatidão percentual
exatidao_percentual = (total_validos / total_dados) * 100 if total_dados > 0 else 0

# Dashboard
st.title("Dashboard NUPAM 2024")

# Sidebar para navegação entre abas
st.sidebar.title("Navegação")
aba = st.sidebar.selectbox("Escolha uma aba:", ["Resumo", "Análises de IA"])

# Resumo Geral
if aba == "Resumo":
    st.subheader("Resumo Geral")
    st.write(f"Total de processos atendidos em 2024: **{total_processos}**")
    st.write(f"Média de dias que o procedimento passou no NUPAM: **{media_duracao:.2f}** dias")
    st.write(f"Total de peças produzidas: **{total_pecas}**")
    st.write(f"Total de tipos de peças: **{total_tipos_pecas}**")

    # Indicadores de Exatidão e Margem de Erro
    st.subheader("Indicadores de Qualidade dos Dados")
    st.write(f"Total de dados: **{total_dados}**")
    st.write(f"Dados faltantes: **{dados_faltantes}**")
    st.write(f"Dados incorretos: **{dados_incorretos}**")
    st.write(f"Total de dados válidos: **{total_validos}**")
    st.write(f"Exatidão percentual: **{exatidao_percentual:.2f}%**")
    st.write(f"Margem de erro: **±{margem_erro:.2f}** dados")
    st.markdown("""
    **Explicação:**
    - **Total de dados:** Número total de registros no DataFrame.
    - **Dados faltantes:** Número total de valores faltantes (NaN) em todas as colunas.
    - **Dados incorretos:** Número de valores faltantes (NaN) na coluna 'Duração'.
    - **Total de dados válidos:** Número de linhas onde a coluna 'Duração' não é NaN.
    - **Exatidão percentual:** Percentual de dados válidos em relação ao total de dados.
    - **Margem de erro:** 5% do número total de dados válidos.
    """)

    # Resumo Estatístico da Duração dos Processos
    st.subheader("Resumo Estatístico da Duração dos Processos")
    resumo_estatistico = {
        "Contagem": df_2024['Duração'].count(),
        "Média": df_2024['Duração'].mean(),
        "Mediana": df_2024['Duração'].median(),
        "Desvio Padrão": df_2024['Duração'].std(),
        "Mínimo": df_2024['Duração'].min(),
        "Máximo": df_2024['Duração'].max(),
        "25º Percentil": df_2024['Duração'].quantile(0.25),
        "50º Percentil": df_2024['Duração'].quantile(0.50),
        "75º Percentil": df_2024['Duração'].quantile(0.75),
        "Total de Processos": total_processos
    }

    # Criar um DataFrame para exibir
    resumo_df = pd.DataFrame(resumo_estatistico, index=[0])
    st.table(resumo_df)
    st.markdown("""
    **Explicação:**
    - **Contagem:** Número total de processos com duração válida.
    - **Média:** Média da duração dos processos.
    - **Mediana:** Valor central da duração dos processos.
    - **Desvio Padrão:** Medida da dispersão dos dados em torno da média.
    - **Mínimo:** Menor valor de duração.
    - **Máximo:** Maior valor de duração.
    - **25º Percentil:** Valor abaixo do qual 25% dos dados se encontram.
    - **50º Percentil:** Valor abaixo do qual 50% dos dados se encontram (mesmo que a mediana).
    - **75º Percentil:** Valor abaixo do qual 75% dos dados se encontram.
    - **Total de Processos:** Número total de processos.
    """)

    # Gráfico de Boxplot interativo
    st.subheader("Distribuição da Duração dos Processos")
    fig_box = px.box(df_2024, x='Duração', title="Boxplot da Duração dos Processos")
    st.plotly_chart(fig_box)
    st.markdown("""
    **Explicação:**
    - **Boxplot:** Gráfico que mostra a distribuição dos dados, incluindo a mediana, os quartis e os outliers.
    """)

    # Gráfico: Processos por Cidade interativo
    st.subheader("Gráfico de Processos por Cidade")
    fig1 = px.bar(processos_por_cidade_full, x='Total de Processos', y='Município', orientation='h',
                  title="Total de Processos por Cidade", color='Total de Processos', 
                  color_continuous_scale='Blues')

    # Atualiza o eixo y para refletir corretamente
    fig1.update_yaxes(title_text="Município")
    st.plotly_chart(fig1)
    st.markdown("""
    **Explicação:**
    - **Gráfico de barras:** Mostra a distribuição geográfica dos processos, com o total de processos em cada cidade.
    """)

    # Gráfico: Processos por Mês interativo
    st.subheader("Gráfico de Processos por Mês")
    fig2 = px.bar(processos_por_mes_sorted, x='Mês', y='Total de Processos', title="Total de Processos por Mês",
                  color='Total de Processos', color_continuous_scale='Reds')
    st.plotly_chart(fig2)
    st.markdown("""
    **Explicação:**
    - **Gráfico de barras:** Mostra a distribuição temporal dos processos, com o total de processos em cada mês.
    """)
    
    # Tabela: Processos por Mês ordenada
    st.subheader("Tabela de Processos por Mês")
    st.table(processos_por_mes_sorted[['Mês', 'Total de Processos']])

    # Tabela de peças
    st.subheader("Tabela de Peças Elaboradas")
    pecas = pecas.sort_values(by='Quantidade', ascending=False)
    st.dataframe(pecas)
    st.markdown("""
    **Explicação:**
    - **Tabela de peças:** Lista os tipos de peças elaboradas e a quantidade de cada tipo.
    """)

# Análises de IA
if aba == "Análises de IA":
    st.subheader("Análises de IA e Aprendizado de Máquina")

    # Regressão Linear
    st.markdown("### Análise de Tendências com Regressão Linear")
    st.markdown("""
    **Explicação:**
    - **Regressão Linear:** Modelo que tenta encontrar uma relação linear entre o índice dos processos e a duração.
    - **Linha de Tendência:** Representa a tendência geral dos dados.
    """)
    X = np.array(range(len(df_2024))).reshape(-1, 1)
    y = df_2024['Duração'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Plotando a linha de tendência
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Dados')
    plt.plot(X, y_pred, color='red', label='Linha de Tendência')
    plt.title('Tendência da Duração dos Processos')
    plt.xlabel('Índice dos-Processos')
    plt.ylabel('Duração (dias)')
    plt.legend()
    st.pyplot(plt)

    # Análise de Cluster com KMeans
    st.markdown("### Análise de Clustering com KMeans")
    st.markdown("""
    **Explicação:**
    - **KMeans:** Algoritmo de clustering que agrupa os dados em clusters com base na duração.
    - **Clusters Identificados:** Mostra os clusters formados e a duração média em cada cluster.
    """)
    kmeans = KMeans(n_clusters=3)
    df_2024['Cluster'] = kmeans.fit_predict(df_2024[['Duração']])
    st.write("Clusters Identificados:")
    st.dataframe(df_2024[['Duração', 'Cluster']])

    # Visualização dos Clusters
    plt.figure(figsize=(10, 5))
    plt.scatter(df_2024['Duração'], df_2024['Cluster'], c=df_2024['Cluster'], cmap='viridis')
    plt.title('Clusters da Duração dos Processos')
    plt.xlabel('Duração (dias)')
    plt.ylabel('Cluster')
    st.pyplot(plt)

    # Previsão de Séries Temporais com ARIMA
    st.markdown("### Previsão de Séries Temporais com ARIMA")
    st.markdown("""
    **Explicação:**
    - **ARIMA:** Modelo de séries temporais que prevê a duração dos processos com base nos dados históricos.
    - **Previsão:** Mostra a previsão da duração dos processos para os próximos 5 dias.
    """)
    df_2024.set_index('Data de entrada', inplace=True)
    model_arima = ARIMA(df_2024['Duração'], order=(5, 1, 0))
    model_fit = model_arima.fit()
    forecast = model_fit.forecast(steps=5)
    st.line_chart(forecast)

    # Neural Network Analysis
    st.markdown("### Análise de Previsão com Redes Neurais")
    st.markdown("""
    **Explicação:**
    - **Redes Neurais:** Modelo de aprendizado de máquina que prevê a duração dos processos com base nos dados históricos.
    - **Previsões:** Mostra as previsões da duração dos processos usando a rede neural.
    """)
    # Prepare data for neural network
    X_nn = np.array(range(len(df_2024))).reshape(-1, 1)
    y_nn = df_2024['Duração'].values

    # Normalize data
    X_nn = X_nn / np.max(X_nn)
    y_nn = y_nn / np.max(y_nn)

    # Define the neural network model
    model_nn = Sequential()
    model_nn.add(Dense(64, input_dim=1, activation='relu'))
    model_nn.add(Dense(32, activation='relu'))
    model_nn.add(Dense(1, activation='linear'))

    # Compile the model
    model_nn.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_nn.fit(X_nn, y_nn, epochs=100, verbose=0)

    # Make predictions
    y_pred_nn = model_nn.predict(X_nn)

    # Plot the neural network predictions
    plt.figure(figsize=(10, 5))
    plt.scatter(X_nn, y_nn, color='blue', label='Dados')
    plt.plot(X_nn, y_pred_nn, color='red', label='Previsões da Rede Neural')
    plt.title('Previsão da Duração dos Processos com Redes Neurais')
    plt.xlabel('Índice dos Processos (normalizado)')
    plt.ylabel('Duração (normalizado)')
    plt.legend()
    st.pyplot(plt)
