# FUNDAMENTOS E WORKFLOW DE MACHINE LEARNING

## O Problema: Predição de Preços de Casas

Imagine que você trabalha para uma imobiliária em Ames, Iowa. Você tem dados de 1460 casas vendidas com 79 características cada (tamanho do lote, número de quartos, ano de construção, etc.) e precisa prever o preço de novas casas.

### Pipeline Básico

```python
# 1. CARREGAR DADOS
X = pd.read_csv('train.csv', index_col='Id')
y = X.SalePrice  # Target (o que queremos prever)
X.drop(['SalePrice'], axis=1, inplace=True)

# 2. DIVIDIR DADOS
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, 
    train_size=0.8,  # 80% para treino
    test_size=0.2,   # 20% para validação
    random_state=0   # Reprodutibilidade
)
```

#### Por que dividir os dados?

- **TREINO (80%)**: O modelo aprende padrões aqui
- **VALIDAÇÃO (20%)**: Testamos se o modelo generaliza bem

**ANALOGIA**: É como estudar para uma prova
- **Treino**: Exercícios que você pratica
- **Validação**: Simulado (prova diferente para testar o aprendizado)
- **Teste**: Prova real (dados nunca vistos)

---

## Métricas de Avaliação: MAE (Mean Absolute Error)

```python
from sklearn.metrics import mean_absolute_error

# Predições
preds = model.predict(X_valid)

# Calcular erro
mae = mean_absolute_error(y_valid, preds)
print(f"MAE: ${mae:.2f}")
```

### O que é MAE?

**Exemplo prático:**

| Casa | Preço Real | Predição | Erro |
|------|-----------|----------|------|
| Casa 1 | $200,000 | $195,000 | $5,000 |
| Casa 2 | $150,000 | $160,000 | $10,000 |
| Casa 3 | $300,000 | $295,000 | $5,000 |

**MAE = (5,000 + 10,000 + 5,000) / 3 = $6,666**

✅ **Menor MAE = Melhor Modelo**

---

## Random Forest: Seu Primeiro Modelo

```python
from sklearn.ensemble import RandomForestRegressor

# Criar modelo
model = RandomForestRegressor(
    n_estimators=100,  # 100 árvores
    random_state=0
)

# Treinar
model.fit(X_train, y_train)

# Prever
predictions = model.predict(X_valid)
```

### Como funciona Random Forest?

**ANALOGIA**: Conselho de especialistas

Imagine que você quer avaliar o preço de uma casa:

- **Especialista 1 (Árvore 1)**: Foca em localização
- **Especialista 2 (Árvore 2)**: Foca em tamanho
- **Especialista 3 (Árvore 3)**: Foca em idade da casa
- ... (97 especialistas mais)

Cada árvore dá uma estimativa, o Random Forest tira a média de todas.

### Comparando Modelos

```python
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
# MAE: 24,015

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
# MAE: 23,740

model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
# MAE: 23,528 ✅ MELHOR

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
# MAE: 23,996

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
# MAE: 23,706

# CONCLUSÃO: Model_3 com criterion='absolute_error' teve melhor performance.
```

---

## TRATAMENTO DE MISSING VALUES

```python
# Investigando missing values
print(X_train.shape)  # (1168, 36)

missing_val_count = X_train.isnull().sum()
print(missing_val_count[missing_val_count > 0])
```

**Output:**
```
LotFrontage    212  (18% dos dados)
MasVnrArea       6  (0.5% dos dados)
GarageYrBlt     58  (5% dos dados)
```

### APPROACH 1: Remover Colunas com Missing Values

```python
# Identificar colunas com missing
cols_with_missing = [col for col in X_train.columns 
                     if X_train[col].isna().any()]

# Remover essas colunas
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Treinar e avaliar
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
# MAE: 17,837.82
```

**✅ Vantagens:**
- Simples e rápido
- Não introduz ruído

**❌ Desvantagens:**
- Perde informação valiosa
- No nosso caso: perdemos 3 colunas inteiras!

### APPROACH 2: Imputação (Preenchimento com Média)

```python
from sklearn.impute import SimpleImputer

# Criar imputer
my_imputer = SimpleImputer(strategy='mean')

# Fit no treino e transform em ambos
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Restaurar nomes das colunas
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# MAE: 18,062.89
```

#### O que aconteceu aqui?

**LotFrontage (frente do lote):**
- Casa 818: Missing → Preenchido com 70.5 (média)
- **Problema**: A casa 818 pode ter um lote muito diferente da média!

```python
SimpleImputer(strategy='mean')      # Média (padrão)
SimpleImputer(strategy='median')    # Mediana (mais robusta a outliers)
SimpleImputer(strategy='most_frequent')  # Moda
SimpleImputer(strategy='constant', fill_value=0)  # Valor fixo
```

### APPROACH 3: Imputação com Mediana (Melhor para Outliers)

```python
final_imputer = SimpleImputer(strategy='median')
final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

# Este foi o escolhido para submissão final!
```

---

## ENCODING DE VARIÁVEIS CATEGÓRICAS

### APPROACH 1: Drop Categorical Variables

```python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
# MAE: 17,837.82
```

Mesma performance de antes porque já tínhamos removido missing values (que coincidentemente estavam em colunas numéricas).

### APPROACH 2: Ordinal Encoding

#### Conceito

Transforma categorias em números ordenados:

```python
MSZoning:
'RL' → 0
'RM' → 1
'C' → 2
'FV' → 3
'RH' → 4
```

#### Problema: Categorias não vistas

```python
print("Unique values in 'Condition2' column in training data:", 
      X_train['Condition2'].unique())
# ['Norm' 'PosA' 'Feedr' 'PosN' 'Artery' 'RRAe']

print("Unique values in 'Condition2' column in validation data:", 
      X_valid['Condition2'].unique())
# ['Norm' 'RRAn' 'RRNn' 'Artery' 'Feedr' 'PosN']
```

**'RRAn' e 'RRNn' aparecem na validação mas NÃO no treino!**

❌ Encoder vai dar ERRO porque não sabe o que fazer com elas.

**Solução: Identificar colunas problemáticas**

```python
# Colunas onde validação é subconjunto do treino
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]

# Colunas problemáticas
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print('Categorical columns that will be ordinal encoded:', good_label_cols)
# 24 colunas OK

print('Categorical columns that will be dropped:', bad_label_cols)
# ['Condition2', 'RoofMatl', 'Functional']
```

#### Implementação

```python
from sklearn.preprocessing import OrdinalEncoder

# Drop colunas ruins
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Aplicar ordinal encoding
ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# MAE: 17,098.01 ✅ MELHOR RESULTADO ATÉ AGORA!
```

### APPROACH 3: One-Hot Encoding

#### Conceito

Cria uma coluna binária para cada categoria:

```python
MSZoning original:
['RL', 'RM', 'RL', 'RL', 'RM']

One-Hot Encoded (5 novas colunas):
MSZoning_C   MSZoning_FV   MSZoning_RH   MSZoning_RL   MSZoning_RM
     0            0             0             1             0
     0            0             0             0             1
     0            0             0             1             0
     0            0             0             1             0
     0            0             0             0             1
```

#### Cardinalidade: O Conceito Crítico

```python
# Contando valores únicos por coluna
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
sorted(d.items(), key=lambda x: x[1])
```

**Output:**
```
('Street', 2)           ← BAIXA cardinalidade ✅ One-Hot
('Utilities', 2)
('LandSlope', 3)
...
('Exterior1st', 15)     ← ALTA cardinalidade ❌ Ordinal ou Drop
('Exterior2nd', 16)
('Neighborhood', 25)
```

#### Exemplo Numérico de Explosão de Features

**Dataset: 10,000 linhas, coluna com 100 categorias únicas**

**One-Hot Encoding:**
- Adiciona: 10,000 × 100 = 1,000,000 entradas
- Dataset original: 10,000 × 1 = 10,000
- **Aumento de 100x!**

**Ordinal Encoding:**
- Adiciona: 0 entradas (substitui a coluna original)
- Mantém mesmo tamanho

#### Implementação

```python
from sklearn.preprocessing import OneHotEncoder

# Apenas baixa cardinalidade (< 10 valores únicos)
low_cardinality_cols = [col for col in object_cols 
                        if X_train[col].nunique() < 10]
# 24 colunas

high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))
# ['Exterior1st', 'Neighborhood', 'Exterior2nd']

# Aplicar One-Hot
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# Restaurar índices
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remover colunas categóricas originais
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Juntar tudo
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
# MAE: 17,525.34
```

---

## PIPELINES: ORGANIZANDO O CAOS

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Identificar colunas
numerical_cols = [col for col in X_train.columns 
                  if X_train[col].dtype in ['int64', 'float64']]

categorical_cols = [col for col in X_train.columns 
                    if X_train[col].dtype == 'object']

# PASSO 1: Preprocessamento numérico
numerical_transformer = SimpleImputer(strategy='constant')

# PASSO 2: Preprocessamento categórico
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# PASSO 3: Combinar preprocessadores
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# PASSO 4: Pipeline completo
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])

# USAR: Simples como um modelo normal!
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print(f"MAE: {mae}")
# MAE: 17,540.47
```

---

## CROSS-VALIDATION: VALIDAÇÃO ROBUSTA

### Split Único

```python
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
```

**Problema**: E se a divisão foi "sortuda"?

**ANALOGIA**: Imagine estudar para uma prova:
- Você faz **1 simulado**
- Tira nota 9
- Na prova real tira 6 😱

**Por quê?** O simulado não representou bem a dificuldade real!

### Solução: K-Fold Cross-Validation

#### Como Funciona (K=5)

```
Dataset completo (100 linhas):

FOLD 1: [VALID][TRAIN][TRAIN][TRAIN][TRAIN]
FOLD 2: [TRAIN][VALID][TRAIN][TRAIN][TRAIN]
FOLD 3: [TRAIN][TRAIN][VALID][TRAIN][TRAIN]
FOLD 4: [TRAIN][TRAIN][TRAIN][VALID][TRAIN]
FOLD 5: [TRAIN][TRAIN][TRAIN][TRAIN][VALID]

Resultado final = MÉDIA dos 5 MAEs
```

#### Implementação

```python
from sklearn.model_selection import cross_val_score

# Criar pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Cross-validation com 5 folds
scores = -1 * cross_val_score(
    my_pipeline, 
    X, y,
    cv=5,  # 5 folds
    scoring='neg_mean_absolute_error'  # Métrica
)

print(f"Average MAE: {scores.mean()}")
print(f"Std Dev: {scores.std()}")
```

**Output:**
```
Average MAE: 18,276.41
Std Dev: 645.23
```

### Tunando Hiperparâmetros

```python
def get_score(n_estimators):
    """Retorna MAE médio com cross-validation"""
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                   cv=3,
                                   scoring='neg_mean_absolute_error')
    return scores.mean()

# Testar diferentes valores
results = {}
for i in range(1, 9):
    results[50*i] = get_score(50*i)

print(results)
```

**Resultados:**
```python
{
    50: 18276.41,
    100: 18100.23,
    150: 17950.67,
    200: 17900.34,  # ✅ MELHOR
    250: 17920.12,
    300: 17935.89,
    350: 17945.23,
    400: 17950.67
}
```

#### Visualização

```python
import matplotlib.pyplot as plt

plt.plot(list(results.keys()), list(results.values()))
plt.xlabel('n_estimators')
plt.ylabel('MAE')
plt.title('Hyperparameter Tuning')
plt.show()
```

**Resultado**: Curva em U invertido
- Poucos estimadores: Underfitting
- **200 estimadores: Sweet spot** ✅
- Muitos estimadores: Overfitting (ligeiro) + tempo de treino

---

## XGBOOST: O CAMPEÃO DE KAGGLE

### O que é Gradient Boosting?

#### Random Forest vs XGBoost

**RANDOM FOREST (Bagging):**
```
100 árvores INDEPENDENTES:
Árvore1: Predição 150k
Árvore2: Predição 160k
Árvore3: Predição 155k
...
RESULTADO = MÉDIA = 155k
```

**XGBOOST (Boosting):**
```
Árvores SEQUENCIAIS que corrigem erros:

Árvore1: Predição 150k (Erro: -10k)
Árvore2: Foca em corrigir o -10k → +8k
Árvore3: Foca em corrigir o -2k restante → +1.5k
Árvore4: Foca em corrigir o -0.5k → +0.3k
...
RESULTADO = 150k + 8k + 1.5k + 0.3k = ~160k (real)
```

**ANALOGIA: Estudando para Prova**

**Random Forest**: 100 alunos estudam sozinhos e cada um faz uma tentativa independente. Você tira a média das respostas.

**XGBoost**:
- Aluno 1 faz a prova (erra algumas questões)
- Aluno 2 estuda especificamente os erros do Aluno 1
- Aluno 3 estuda os erros que o Aluno 2 ainda deixou
- ...

**Resultado**: Cada aluno seguinte fica melhor nos pontos fracos dos anteriores!

### Implementação Básica

```python
from xgboost import XGBRegressor

# MODEL 1: Baseline (parâmetros default)
my_model_1 = XGBRegressor(random_state=0)
my_model_1.fit(X_train, y_train)

predictions_1 = my_model_1.predict(X_valid)
mae_1 = mean_absolute_error(predictions_1, y_valid)
print(f"MAE: {mae_1}")
# MAE: 18,161.82
```

### Otimizando XGBoost

#### Principais Hiperparâmetros

```python
# MODEL 2: Otimizado
my_model_2 = XGBRegressor(
    n_estimators=1000,    # Mais árvores
    learning_rate=0.03,   # Aprendizado mais lento e cuidadoso
    random_state=0
)

my_model_2.fit(X_train, y_train)
predictions_2 = my_model_2.predict(X_valid)
mae_2 = mean_absolute_error(predictions_2, y_valid)
print(f"MAE: {mae_2}")
# MAE: 17,281.12 ✅ MELHOR RESULTADO DE TODOS!
```

### Entendendo os Parâmetros

**1. n_estimators (número de árvores):**
```
n_estimators=50:  MAE 18,500
n_estimators=100: MAE 18,200
n_estimators=500: MAE 17,500
n_estimators=1000: MAE 17,281 ✅
n_estimators=5000: MAE 17,300 (overfitting)
```
➡️ **Mais árvores = melhor** (até certo ponto)

**2. learning_rate (taxa de aprendizado):**
```
learning_rate=0.1 (default):  Aprende rápido mas pode pular o ótimo
learning_rate=0.03:           Aprende devagar mas com mais precisão ✅
learning_rate=0.01:           Aprende muito devagar (precisa de mais árvores)
```

**ANALOGIA:**
- **learning_rate alto**: Dar passos grandes ao procurar chaves no escuro (rápido mas pode passar direto)
- **learning_rate baixo**: Dar passos pequenos (lento mas encontra com precisão)

**Outros parâmetros importantes:**

```python
XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,           # Profundidade máxima de cada árvore
    min_child_weight=1,    # Mínimo de amostras por folha
    subsample=0.8,         # Usa 80% dos dados em cada árvore (previne overfitting)
    colsample_bytree=0.8,  # Usa 80% das features em cada árvore
    random_state=0
)
```

### Experimentando: "Breaking the Model"

```python
# MODEL 3: Propositalmente ruim
my_model_3 = XGBRegressor(n_estimators=1)  # Apenas 1 árvore!

my_model_3.fit(X_train, y_train)
predictions_3 = my_model_3.predict(X_valid)
mae_3 = mean_absolute_error(predictions_3, y_valid)
print(f"MAE: {mae_3}")
# MAE: 42,678.81 😱
```

---

## DATA LEAKAGE: O ERRO SILENCIOSO

### O que é Data Leakage?

**Data Leakage** ocorre quando informações do conjunto de validação/teste "vazam" para o treinamento, fazendo o modelo parecer muito melhor do que realmente é.

**ANALOGIA**: É como estudar com as respostas da prova!
- Seu desempenho no simulado será perfeito ✅
- Mas na prova real você vai mal ❌

### Exemplo Clássico: O Caso das Pneumonias

**Cenário Real (baseado em caso verídico):**

Um hospital treinou um modelo para detectar pneumonia em raios-X com 95% de acurácia (impressionante!). Mas na prática, o modelo falhou completamente.

**O que aconteceu?**

```python
# O que eles ACHARAM que fizeram:
X_train, X_valid = train_test_split(xray_images)
model.fit(X_train)
accuracy = model.score(X_valid)  # 95%! 🎉

# O que REALMENTE aconteceu:
# Todas as imagens de pneumonia tinham um "marker" no canto
# (identificador do equipamento portátil usado em pacientes graves)
# O modelo aprendeu: "marker no canto = pneumonia" 🤦
```

Quando testaram em outro hospital (sem o marker), o modelo foi terrível!

---

### Tipos de Data Leakage

#### 1. Target Leakage

**Definição**: Usar variáveis que não estarão disponíveis na predição real.

**Exemplo 1: Predição de Inadimplência**

```python
# ❌ ERRADO - Target Leakage
features = ['renda', 'idade', 'divida_atual', 'pagamento_atrasado']
target = 'inadimplente'

# "pagamento_atrasado" só existe DEPOIS da pessoa ficar inadimplente!
# É como prever se alguém vai se atrasar... usando o fato de que ela se atrasou 🤦
```

```python
# ✅ CORRETO
features = ['renda', 'idade', 'divida_atual', 'historico_credito']
target = 'inadimplente'

# Agora só usamos informações disponíveis ANTES da inadimplência
```

**Exemplo 2: Predição de Doenças**

```python
# Dataset de pacientes
paciente_id | sintomas | exames_realizados | tem_doenca
001         | febre    | 5                 | sim
002         | tosse    | 1                 | nao
003         | dor      | 8                 | sim

# ❌ ERRADO: Usar 'exames_realizados' como feature
# Médicos pedem MAIS exames quando suspeitam da doença!
# O modelo aprende: "muitos exames = doente" (não é causalidade!)
```

**Como Detectar:**

```python
# Se remover uma feature e a performance DESPENCAR drasticamente,
# pode ser target leakage!

model_com_feature.score()    # 98%
model_sem_feature.score()    # 65%
# Diferença suspeita! 🚨
```

---

#### 2. Train-Test Contamination

**Definição**: Usar informações do conjunto de teste durante o pré-processamento.

**Exemplo ERRADO:**

```python
# ❌ ERRO CLÁSSICO - Fazer pré-processamento ANTES de dividir os dados

from sklearn.preprocessing import StandardScaler

# 1. Normalizar TODOS os dados
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)  # 🚨 VAZAMENTO AQUI!

# 2. Depois dividir
X_train, X_valid = train_test_split(X_normalized)

# 3. Treinar
model.fit(X_train, y_train)
```

**Por que está errado?**

```python
# O scaler calculou média e desvio usando TODO o dataset:
# mean = 50, std = 10

# Isso significa que o conjunto de TREINO "viu" informações do conjunto de VALIDAÇÃO!

# Exemplo numérico:
Dataset completo: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
Média: 55

Treino: [10, 20, 30, 40, 50]  # Normalizado usando média 55 (que inclui dados de validação!)
Valid:  [60, 70, 80, 90, 100] # Normalizado usando média 55 (correto aqui)

# A média 55 foi "contaminada" pelos dados de validação!
```

**Exemplo CORRETO:**

```python
# ✅ CORRETO - Dividir PRIMEIRO, processar DEPOIS

# 1. Dividir primeiro
X_train, X_valid = train_test_split(X)

# 2. Fit no treino, transform em ambos
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)  # Aprende só do treino
X_valid_normalized = scaler.transform(X_valid)      # Aplica o aprendido

# 3. Treinar
model.fit(X_train_normalized, y_train)
```

**Comparação Visual:**

```python
# ❌ ERRADO
[Todos os dados] 
    ↓ (fit_transform)
[Dados normalizados]
    ↓ (split)
[Treino] [Validação]  # Validação influenciou a normalização do treino!

# ✅ CORRETO
[Todos os dados]
    ↓ (split)
[Treino] [Validação]
    ↓ (fit_transform no treino)
[Treino normalizado] [Validação transform]  # Validação não influenciou!
```

---

### Exemplo Prático: Preços de Casas com Leakage

**Cenário**: Você quer prever preços de casas e tem uma feature `preco_medio_bairro`.

```python
# Dataset
casa_id | quartos | bairro    | preco_medio_bairro | preco
001     | 3       | Centro    | 250000             | 240000
002     | 2       | Suburbio  | 180000             | 185000
003     | 4       | Centro    | 250000             | 260000

# ❌ PROBLEMA: 'preco_medio_bairro' foi calculado INCLUINDO os preços
# que você está tentando prever!

# Como foi calculado (ERRADO):
df['preco_medio_bairro'] = df.groupby('bairro')['preco'].transform('mean')
# Isso inclui o preço da própria casa no cálculo da média!
```

**Simulação do Impacto:**

```python
# Com leakage
model_with_leakage = RandomForestRegressor()
model_with_leakage.fit(X_train, y_train)
print(model_with_leakage.score(X_valid))
# R² = 0.95 (parece incrível!)

# Sem a feature problemática
X_train_clean = X_train.drop(['preco_medio_bairro'], axis=1)
X_valid_clean = X_valid.drop(['preco_medio_bairro'], axis=1)
model_clean = RandomForestRegressor()
model_clean.fit(X_train_clean, y_train)
print(model_clean.score(X_valid_clean))
# R² = 0.78 (resultado realista)

# Produção (dados novos, sem a feature)
# R² = 0.65 😱 Performance caiu muito!
```

**Solução Correta:**

```python
# ✅ CORRETO: Calcular média apenas no conjunto de treino

# 1. Split primeiro
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# 2. Calcular média apenas no treino
bairro_means = X_train.groupby('bairro')['preco'].mean()

# 3. Aplicar nos dois conjuntos
X_train['preco_medio_bairro'] = X_train['bairro'].map(bairro_means)
X_valid['preco_medio_bairro'] = X_valid['bairro'].map(bairro_means)

# Se um bairro aparecer só na validação, use a média global do treino
global_mean = y_train.mean()
X_valid['preco_medio_bairro'].fillna(global_mean, inplace=True)
```

---

### Como Detectar Data Leakage

#### 1. Performance Suspeita

```python
# 🚨 SINAIS DE ALERTA

# Modelo MUITO bom na validação
validation_score = 0.99  # Perfeito demais!

# Performance cai drasticamente em produção
production_score = 0.65  # Caiu muito! 😱

# Uma feature tem importância absurda
feature_importance = {
    'area': 0.15,
    'quartos': 0.12,
    'mysterious_feature': 0.60  # 🚨 Suspeito!
}
```

#### 2. Teste de Sanidade

```python
# Remover features uma por vez e ver o impacto

baseline_mae = 17000

for col in X.columns:
    X_temp = X.drop(columns=[col])
    model.fit(X_train_temp, y_train)
    mae = mean_absolute_error(y_valid, model.predict(X_valid_temp))
    
    if baseline_mae - mae > 5000:  # Caiu muito!
        print(f"🚨 SUSPEITA: {col} pode ter leakage")
        print(f"   MAE sem ela: {mae}")
        print(f"   MAE com ela: {baseline_mae}")
```

#### 3. Checklist de Prevenção

```python
# ✅ CHECKLIST ANTI-LEAKAGE

# 1. Cada feature está disponível no momento da predição?
#    - "status_do_pedido" ao prever se pedido será cancelado? ❌
#    - "histórico_de_compras" ao prever próxima compra? ✅

# 2. Split foi feito ANTES de qualquer pré-processamento?
X_train, X_valid = train_test_split(X)  # Primeiro!
scaler.fit(X_train)  # Depois!

# 3. Features agregadas foram calculadas apenas no treino?
bairro_stats = X_train.groupby('bairro').mean()  # Só treino!

# 4. Validação temporal está correta? (séries temporais)
# Treino: 2020-2022
# Valid:  2023  # Não usar dados futuros no treino!

# 5. Features criadas após o evento que você prevê?
#    - Prever morte: usar "data_do_óbito" ❌
#    - Prever churn: usar "motivo_do_cancelamento" ❌
```

---

### Exemplo Completo: Pipeline Sem Leakage

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# 1. Split PRIMEIRO (sempre!)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Identificar tipos de colunas
numeric_features = ['area', 'quartos', 'idade_casa']
categorical_features = ['bairro', 'tipo']

# 3. Criar transformers (vão fit apenas no treino!)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Fit só no treino automaticamente!
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. Combinar
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 5. Pipeline final
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6. Fit (preprocessor aprende APENAS do treino)
model.fit(X_train, y_train)

# 7. Predict (preprocessor usa o que aprendeu do treino)
preds = model.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)

print(f"MAE: {mae}")  # Resultado confiável! ✅
```

---

### Data Leakage em Séries Temporais

**Problema Especial**: Ordem temporal importa!

```python
# ❌ ERRADO - Treinar com dados futuros
dados = pd.DataFrame({
    'data': ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05'],
    'vendas': [100, 120, 110, 130, 125]
})

X_train, X_valid = train_test_split(dados, test_size=0.2)
# Pode pegar 2020-05 no treino e 2020-02 na validação! ❌
```

```python
# ✅ CORRETO - Respeitar ordem temporal
train_size = int(len(dados) * 0.8)

train = dados[:train_size]   # Primeiros 80%
valid = dados[train_size:]   # Últimos 20%

# Treino: 2020-01 a 2020-04
# Valid:  2020-05
# Nunca treinar com dados do futuro!
```

---

### Resumo Final: Data Leakage

| Tipo | Descrição | Como Evitar |
|------|-----------|-------------|
| **Target Leakage** | Usar features que não existem na predição | Questione: "Isso estará disponível quando eu prever?" |
| **Train-Test Contamination** | Pré-processar antes de split | SEMPRE: Split → Fit → Transform |
| **Temporal Leakage** | Usar dados futuros | Respeitar ordem temporal |
| **Aggregation Leakage** | Média inclui o próprio target | Calcular agregações só no treino |

**Regra de Ouro**: 🏆
> **Se parece bom demais para ser verdade, provavelmente é leakage!**

---

## Resumo de Resultados

| Técnica | MAE | Status |
|---------|-----|--------|
| Random Forest (básico) | 24,015 | Baseline |
| Random Forest (otimizado) | 23,528 | Melhor RF |
| Drop Missing Values | 17,837 | OK |
| Imputação Média | 18,062 | Pior que drop |
| Ordinal Encoding | 17,098 | Ótimo |
| One-Hot Encoding | 17,525 | Bom |
| Pipeline Completo | 17,540 | Organizado |
| Cross-Validation | 17,900 | Robusto |
| **XGBoost Otimizado** | **17,281** | **🏆 MELHOR** |

---

## Conclusões e Próximos Passos

### O que aprendemos:

1. **Sempre divida seus dados** (treino/validação/teste)
2. **Missing values**: Imputação geralmente é melhor que remover
3. **Encoding**: Escolha baseado na cardinalidade
4. **Pipelines**: Organizam e evitam erros
5. **Cross-validation**: Validação mais robusta
6. **XGBoost**: Geralmente supera Random Forest

### Próximos passos para melhorar:

- Feature Engineering (criar novas features)
- Grid Search / Random Search para hiperparâmetros
- Ensemble de múltiplos modelos
- Análise de importância de features
- Tratamento de outliers
- Normalização/Padronização de features

---

