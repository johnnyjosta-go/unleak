import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("energy.csv")
for lag in (1, 2, 24):
    df[f"load_lag_{lag}"] = df["load"].shift(lag)
df = df.dropna()
X = df.drop(columns=["load"])
y = df["load"]
cv = KFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(Ridge(), X, y, cv=cv))  # expect: UL402
