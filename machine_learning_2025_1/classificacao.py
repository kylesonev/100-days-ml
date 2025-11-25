# %%
from sklearn import naive_bayes
from sklearn import linear_model, tree
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")

df.head()

# %%
df['aprovado'] = df['nota'] > 5
df

# %%
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.title("Cerveja vs Aprovação")
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.grid(True)

# %%
reg = linear_model.LogisticRegression(penalty=None,
                                      fit_intercept=True)

reg.fit(df[['cerveja']], df[['aprovado']])

reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_prob = reg.predict_proba(df[['cerveja']].drop_duplicates())[:, 1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_prob = arvore_full.predict_proba(
    df[['cerveja']].drop_duplicates())[:, 1]

# %%
plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.title("Cerveja vs Aprovação")
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.grid(True)
plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color="tomato")
plt.plot(df['cerveja'].drop_duplicates(), reg_prob, color="red")
plt.legend(["Observação", "Reg Predict", "Reg Proba"])
plt.hlines(0.5, xmin=1, xmax=9, linestyles="--", color="black")

# %%
plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.title("Cerveja vs Aprovação")
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.grid(True)
plt.plot(df['cerveja'].drop_duplicates(), arvore_full_predict, color="green")
plt.plot(df['cerveja'].drop_duplicates(), arvore_full_prob, color="pink")
plt.legend(["Observação",
            "Árvore Predict",
            "Árvore Proba"])
plt.hlines(0.5, xmin=1, xmax=9, linestyles="--", color="black")

# %%
nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_prob = nb.predict_proba(df[['cerveja']].drop_duplicates())[:, 1]

# %%
plt.figure(dpi=400)
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.title("Cerveja vs Aprovação")
plt.xlabel("Cerveja")
plt.ylabel("Aprovado")
plt.grid(True)
plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color="blueviolet")
plt.plot(df['cerveja'].drop_duplicates(), nb_prob, color="darkred")
plt.legend(["Observação",
            "Árvore Predict",
            "Árvore Proba"])
plt.hlines(0.5, xmin=1, xmax=9, linestyles="--", color="black")


# %%
