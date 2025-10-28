## Exemplos das Funções do Pandas
### 1. Criação de Dados
DataFrame
```
# Criar DataFrame simples
pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})

# DataFrame com strings
pd.DataFrame({
    'Bob': ['I liked it.', 'It was awful.'], 
    'Sue': ['Pretty good.', 'Bland.']
})

# DataFrame com índice personalizado
pd.DataFrame({
    'Bob': ['I liked it.', 'It was awful.'], 
    'Sue': ['Pretty good.', 'Bland.']
}, index=['Product A', 'Product B'])
```
Series

```
# Series simples
pd.Series([1, 2, 3, 4, 5])

# Series com índice e nome
pd.Series([30, 35, 40], 
         index=['2015 Sales', '2016 Sales', '2017 Sales'], 
         name='Product A')
```
---

### 2. Leitura de Arquivos

```
# Ler CSV
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")

# Ler CSV com índice específico
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", 
                           index_col=0)

# Verificar dimensões
wine_reviews.shape  # (129971, 14)

# Ver primeiras linhas
wine_reviews.head()
```

---

### 3. Indexação e Seleção
**Acessores Nativos**
```
# Por atributo
reviews.country

# Por indexação
reviews['country']

# Valor específico
reviews['country'][0]  # 'Italy'
```
**iloc (posicional)**
```
# Primeira linha
reviews.iloc[0]

# Primeira coluna (todas as linhas)
reviews.iloc[:, 0]

# Primeiras 3 linhas, primeira coluna
reviews.iloc[:3, 0]

# Linhas 1 e 2, primeira coluna
reviews.iloc[1:3, 0]

# Lista de índices
reviews.iloc[[0, 1, 2], 0]

# Últimas 5 linhas
reviews.iloc[-5:]
```
**loc (por rótulos)**

```
# Valor específico
reviews.loc[0, 'country']  # 'Italy'

# Múltiplas colunas
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

```
**Manipulação de Índice**

```
# Definir nova coluna como índice
reviews.set_index("title")
```
**Seleção Condicional**

```
# Condição simples
reviews.country == 'Italy'

# Filtrar com condição
reviews.loc[reviews.country == 'Italy']

# Múltiplas condições (AND)
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]

# Múltiplas condições (OR)
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]

# Valores em lista
reviews.loc[reviews.country.isin(['Italy', 'France'])]

# Valores não nulos
reviews.loc[reviews.price.notnull()]

# Valores nulos
reviews[pd.isnull(reviews.country)]
```
**Atribuição**

```
# Valor constante
reviews['critic'] = 'everyone'

# Valores iteráveis
reviews['index_backwards'] = range(len(reviews), 0, -1)
```
---
### 4. Funções Resumo

```
# Estatísticas descritivas
reviews.points.describe()
# Saída:
# count    129971.000000
# mean         88.447138
# std           3.039730
# min          80.000000
# 25%          86.000000
# 50%          88.000000
# 75%          91.000000
# max         100.000000

# Descrição de strings
reviews.taster_name.describe()
# Saída:
# count         103727
# unique            19
# top       Roger Voss
# freq           25514

# Média
reviews.points.mean()  # 88.44713820775404

# Valores únicos
reviews.taster_name.unique()
# array(['Kerin O'Keefe', 'Roger Voss', 'Paul Gregutt', ...])

# Contagem de valores
reviews.taster_name.value_counts()
# Roger Voss           25514
# Michael Schachner    15134
# ...

```
---
### 5. Map e Apply
**map()**

```
# Subtrair a média
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean)
# 0        -1.447138
# 1        -1.447138
# ...
```

**apply()**


```
# Aplicar função em linhas
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')
```

**Operadores Vetorizados**

```
# Subtração
reviews.points - review_points_mean

# Concatenação de strings
reviews.country + " - " + reviews.region_1
# 0            Italy - Etna
# 1                     NaN
# ...
```
---

### 6. GroupBy
**Agrupamento Básico**


```
# Contar por grupo
reviews.groupby('points').points.count()
# points
# 80     397
# 81     692
# ...

# Valor mínimo por grupo
reviews.groupby('points').price.min()
# points
# 80      5.0
# 81      5.0
# ...
```

**Apply em Grupos**

```
# Primeiro vinho de cada vinícola
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])
# winery
# 1+1=3        1+1=3 NV Rosé Sparkling (Cava)
# 10 Knots     10 Knots 2010 Viognier (Paso Robles)
# ...
```

**Agrupamento Múltiplo**

```
# Melhor vinho por país e província
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
```


**Agregações Múltiplas**


```
# Várias estatísticas de uma vez
reviews.groupby(['country']).price.agg([len, min, max])
#             len   min    max
# country                     
# Argentina  3800   4.0  230.0
# Armenia       2  14.0   15.0
# ...
```

---

### 7. MultiIndex

```
# Criar MultiIndex
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])

# Verificar tipo do índice
mi = countries_reviewed.index
type(mi)  # pandas.core.indexes.multi.MultiIndex

# Resetar para índice simples
countries_reviewed.reset_index()
```
---
### 8. Ordenação
```
# Ordenar por valores (ascendente)
countries_reviewed.sort_values(by='len')

# Ordenar por valores (descendente)
countries_reviewed.sort_values(by='len', ascending=False)
#     country    province    len
# 392      US  California  36247
# 415      US  Washington   8639
# ...

# Ordenar por índice
countries_reviewed.sort_index()

# Ordenar por múltiplas colunas
countries_reviewed.sort_values(by=['country', 'len'])
```

---


### 9. Tipos de Dados

```
# Tipo de uma coluna
reviews.price.dtype  # dtype('float64')

# Tipos de todas as colunas
reviews.dtypes
# country        object
# description    object
# ...
# variety        object
# winery         object

# Tipo do índice
reviews.index.dtype  # dtype('int64')

# Converter tipo
reviews.points.astype('float64')
# 0         87.0
# 1         87.0
# ...

```
---

### 10. Valores Ausentes

```
# Detectar valores nulos
pd.isnull(reviews.country)

# Filtrar valores nulos
reviews[pd.isnull(reviews.country)]

# Preencher valores ausentes
reviews.region_2.fillna("Unknown")
# 0         Unknown
# 1         Unknown
# ...

# Substituir valores
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
# 0            @kerino
# 1         @vossroger
# ...
```
---

### 11. Renomeação
```
# Renomear colunas
reviews.rename(columns={'points': 'score'})

# Renomear índices
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# Renomear eixos
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```

---

### 12. Combinação de DataFrames
**concat()**

```# Concatenar DataFrames verticalmente
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])

```

**join()**
```
# Preparar índices
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

# Juntar DataFrames
left.join(right, lsuffix='_CAN', rsuffix='_UK')
```

---

### Resumo de Operações Comuns


```
# Exploração inicial
df.shape
df.head()
df.dtypes
df.describe()

# Seleção
df['coluna']
df.iloc[0]
df.loc[0, 'coluna']
df[df.coluna > 5]

# Transformação
df['nova'] = df.coluna.map(lambda x: x * 2)
df.apply(funcao, axis='columns')

# Agregação
df.groupby('coluna').agg(['mean', 'count'])
df.value_counts()

# Limpeza
df.fillna(0)
df.dropna()
df.replace('old', 'new')

# Ordenação
df.sort_values('coluna', ascending=False)

```






