import pandas as pd
from collections import Counter
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Carregar o arquivo CSV
df = pd.read_csv('processos.csv')

# Ajuste para tornar o nome das colunas insensível a maiúsculas/minúsculas
df.columns = [col.strip().upper() for col in df.columns]

# Função para separar e limpar as peças
def separar_pecas(peca_texto):
    if pd.isna(peca_texto):  # Ignora valores NaN
        return []
    pecas = re.split(r'\s*\+\s*', peca_texto.lower())
    return [peca.strip() for peca in pecas if peca.strip()]  # Ignora peças vazias

# Extraindo e padronizando as peças
todas_pecas = []
for texto_pecas in df['PEÇAS ELABORADAS']:  # Acessa agora em maiúsculas
    todas_pecas.extend(separar_pecas(texto_pecas))

# Função para agrupar peças por similaridade
def agrupar_por_similaridade(lista_pecas, limite=85):  # Limite ajustado para maior similaridade
    grupos = []
    usados = set()
    
    for peca in lista_pecas:
        if peca in usados:
            continue
        similares = process.extract(peca, lista_pecas, limit=None, scorer=fuzz.token_sort_ratio)
        grupo_atual = [similar for similar, score in similares if score >= limite]
        usados.update(grupo_atual)
        grupos.append(grupo_atual)
    
    padronizado = {item: grupo[0].capitalize() for grupo in grupos for item in grupo}
    return padronizado

# Agrupar e padronizar as peças
pecas_padronizadas = agrupar_por_similaridade(todas_pecas)
pecas_normalizadas = [pecas_padronizadas[peca] for peca in todas_pecas]

# Contagem das peças padronizadas
contagem_pecas = Counter(pecas_normalizadas)

# Converter a contagem para um DataFrame
df_contagem = pd.DataFrame(contagem_pecas.items(), columns=['Peça', 'Quantidade'])

# Salvar o DataFrame em um arquivo CSV
df_contagem.to_csv('contagem_pecas.csv', index=False)

# Exibir o resultado
for peca, qtd in contagem_pecas.items():
    print(f"{peca}: {qtd}")