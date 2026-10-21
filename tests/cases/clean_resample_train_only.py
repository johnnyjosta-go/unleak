import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv("churn.csv")
train, test = train_test_split(df, test_size=0.2, stratify=df["target"])
maj = train[train["target"] == 0]
mino = train[train["target"] == 1]
mino_up = resample(mino, replace=True, n_samples=len(maj))
train_bal = pd.concat([maj, mino_up])
model = RandomForestClassifier().fit(train_bal.drop(columns="target"), train_bal["target"])
print(model.score(test.drop(columns="target"), test["target"]))
missing = df.isnull().mean()
df = df.drop(columns=missing[missing > 0.4].index)
yearly = pd.concat([df[df["year"] == 2019][:100], df[df["year"] == 2020]])
