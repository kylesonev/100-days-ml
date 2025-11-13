# %%
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")

df.head()

# %%
X = df[["cerveja"]]
y = df["nota"]

# Linear Regression
reg = linear_model.LinearRegression()
reg.fit(X, y)
predict = reg.predict(X.drop_duplicates())


# Decision tree
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
# %%
a, b = reg.intercept_, reg.coef_[0]

print(a, b)

# %%
plt.plot(X["cerveja"], y, 'o')
plt.grid(True)
plt.title("Revelação cerveja vs nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict, '-')
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full, '-')
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2, '-')

plt.legend(['Observado',
            f'y = {a:.2f} + {b:.2f} x',
            'Árvore Full',
            ])

# %%
plt.figure(dpi=400)
tree.plot_tree(arvore_full,
               feature_names=['cerveja'],
               filled=True)

# %%
plt.figure(dpi=400)
tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled=True)
