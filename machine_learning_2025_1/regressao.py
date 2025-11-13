# %%
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")

df.head()
# %%

X = df[["cerveja"]]
y = df["nota"]

reg = linear_model.LinearRegression()
# %%
reg.fit(X, y)
# %%
a, b = reg.intercept_, reg.coef_[0]

print(a, b)

# %%
predict = reg.predict(X.drop_duplicates())
predict
# %%
plt.plot(X["cerveja"], y, 'o')
plt.grid(True)
plt.title("Revelação cerveja vs nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.plot(X.drop_duplicates()['cerveja'], predict, '-')
plt.legend(['Observado', f'y = {a:.2f} + {b:.2f} x'])
# %%
