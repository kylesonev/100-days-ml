# %%
from sklearn import tree
import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")

df

# %%
df["General Jedi encarregado"] = df['General Jedi encarregado'].map({'Yoda': 1,
                                                                     'Shaak Ti': 2,
                                                                     'Obi-Wan Kenobi': 3,
                                                                     'Aayla Secura': 4,
                                                                     'Mace Windu': 5})


# %%
df.columns
X = df[['Massa(em kilos)',
       'Estatura(cm)', 'Tempo de existência(em meses)']]
y = df['Status ']


# %%
model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# %%
tree.plot_tree(model, max_depth=3)
# %%
