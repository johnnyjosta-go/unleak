import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic.csv")
df["Age"] = df["Age"].fillna(df["Age"].mean())  # expect: UL106
df["Fare"].fillna(df["Fare"].median(), inplace=True)  # expect: UL106
print(df.describe())
print(df.mean(axis=1))
train, test = train_test_split(df, test_size=0.2)
