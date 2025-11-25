# %%
from sklearn import tree
import pandas as pd


df = pd.read_csv("data/dados_comunidade.csv")

print(df.info())

# %%
df = df.replace({'Sim': 1, 'Não': 0})
df.head()

# %%

dummy_vars = ['Como conheceu o Téo Me Why?',
              'Quantos cursos acompanhou do Téo Me Why?',
              'Estado que mora atualmente',
              'Área de Formação',
              'Tempo que atua na área de dados',
              'Posição da cadeira (senioridade)']

num_vars = ['Curte games?',
            'Curte futebol?',
            'Curte livros?',
            'Curte jogos de tabuleiro?',
            'Curte jogos de fórmula 1?',
            'Curte jogos de MMA?',
            'Idade'
            ]
# %%
df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
# %%
df_analise['Pessoa Feliz'] = df['Você se considera uma pessoa feliz?'].copy()

# %%
df_analise

# %%
X = df_analise.drop(columns='Pessoa Feliz')
y = df_analise['Pessoa Feliz']
arvore = tree.DecisionTreeClassifier(random_state=42,
                                     min_samples_leaf=5
                                     )
arvore.fit(X, y)


# %%
arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analise[['Pessoa Feliz']]
df_predict['Predict'] = arvore_predict
df_predict

# %%
(df_predict['Pessoa Feliz'] == df_predict['Predict']).mean()
# %%
pd.crosstab(df_predict['Pessoa Feliz'], df_predict['Predict'])


# %%
