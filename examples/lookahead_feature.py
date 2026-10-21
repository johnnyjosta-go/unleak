"""UL401 - a feature that peeks at tomorrow.

Daily prices from a random walk: tomorrow's direction is a coin flip. A
"momentum" feature built with shift(-1) quietly contains the answer.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RULE = "UL401"
TITLE = "Look-ahead feature (random-walk prices)"
METRIC = "direction accuracy"


def data(seed):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000)))
    return pd.DataFrame({"close": close})


def leaky(df):
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["momentum"] = df["close"].pct_change().shift(-1).rolling(3).mean()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna().iloc[:-1]
    X = df.drop(columns=["target", "close"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def honest(df):
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["momentum"] = df["close"].pct_change().rolling(3).mean()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna().iloc[:-1]
    X = df.drop(columns=["target", "close"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    model = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def run(seed=0):
    df = data(seed)
    return leaky(df), honest(df)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
