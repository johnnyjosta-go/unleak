import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv("cancer.csv")
df["label"] = df["diagnosis"].map({"M": 1, "B": 0})  # expect: UL303
X = df.drop(["label", "id"], axis=1)
y = df["label"]
SVC().fit(X, y)
