import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def clean(df):
    df = df.copy()
    df["income"] = df["income"].fillna(df["income"].median())  # expect: UL106
    return df


def load():
    raw = pd.read_csv("loans.csv")
    return raw


data = clean(load())
X = data.drop(columns=["default"])
y = data["default"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
RandomForestClassifier().fit(X_train, y_train)
