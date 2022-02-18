# %%
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

# %%
df = pd.read_csv("insurance_data.csv")
df.head()

# %%
plt.scatter(df.age, df.bought_insurance, marker="+", color="red")

# %%
df.shape

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.10)

# %%
X_test

# %%
from sklearn.linear_model import LogisticRegression

# %%
model = LogisticRegression()

# %%
model.fit(X_train, y_train)

# %%
model.predict(X_test)

# %%
model.score(X_test, y_test)

# %%
model.predict_proba(X_test)

# %% [markdown]
# model.predict throws a ValueError when handling scalar, therefore I converted the value to a 2D array using double parenthesis [[]].

# %%
model.predict([[25]])


