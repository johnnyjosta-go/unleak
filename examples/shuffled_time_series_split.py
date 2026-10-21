"""UL402 - random split on a time series with overlapping targets.

Predict whether a random walk is higher in 10 days (a coin flip by
construction) from lagged prices. A shuffled split puts each test day between
training days whose targets overlap with it; a time split does not.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RULE = "UL402"
TITLE = "Shuffled split on a time series (random walk)"
METRIC = "accuracy (10-day direction)"


def data(seed):
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
    df = pd.DataFrame({"price": price})
    df["target"] = (df["price"].shift(-10) > df["price"]).astype(int)
    return df.iloc[:-10]


def features(df):
    df = df.copy()
    for lag in (1, 5, 20):
        df[f"lag_{lag}"] = df["price"].shift(lag)
    return df.dropna()


def leaky(df, seed):
    df = features(df)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed).fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def honest(df, seed):
    df = features(df)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    X_train, y_train = X_train.iloc[:-10], y_train.iloc[:-10]  # drop targets that reach into the test period
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed).fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def run(seed=0):
    df = data(seed)
    return leaky(df, seed), honest(df, seed)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
