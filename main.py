import pandas as pd
from matplotlib import pyplot as plt
# %matplotlib inline # use when running in jupiter labs

df  = pd.read_csv("insurance_data.csv")
df.head()
print(df.head())

plt.scatter(df.age, df.bought_insurance, marker="+", color="red")