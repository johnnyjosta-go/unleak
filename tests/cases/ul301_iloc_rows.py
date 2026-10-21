import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv").dropna().iloc[:-1]
y = df["label"]
X = df.drop(columns=["id"])  # expect: UL301
LogisticRegression().fit(X, y)
X2 = df.iloc[:, :-1]
LogisticRegression().fit(X2, y)
