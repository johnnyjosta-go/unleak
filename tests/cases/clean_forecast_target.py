import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("prices.csv")
df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
df["ma_10"] = df["close"].rolling(10).mean()
y = df["target"]
X = df.drop(columns=["target"])
LogisticRegression().fit(X[:800], y[:800])
