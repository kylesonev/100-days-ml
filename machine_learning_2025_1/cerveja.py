# %%
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd

df = pd.read_excel("data/dados_cerveja.xlsx")
df

# %%
features = ["temperatura", "copo", "espuma", "cor"]
X = df[features]
y = df["classe"]

# %%
X = X.replace({"mud": 1, "pint": 2, "sim": 1, "não": 0, "clara": 0, "escura": 1})

# %%

model = tree.DecisionTreeClassifier(random_state=42)

model.fit(X, y)

# %%
plt.ion()
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
