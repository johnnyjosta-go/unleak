"""UL303 - the label is a threshold on a column that stays in X.

"High-risk" is defined as more than 90 days late, and days_late is still a
feature. The model learns the threshold, not the risk.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RULE = "UL303"
TITLE = "Label defined from a column that stays a feature"
METRIC = "accuracy"


def data(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    income = rng.normal(50, 15, n)
    debt = rng.normal(20, 8, n)
    risk = 0.04 * debt - 0.03 * income + rng.normal(0, 1, n)
    days_late = np.clip(60 + 40 * risk + rng.normal(0, 25, n), 0, None)
    return pd.DataFrame({"income": income, "debt": debt, "days_late": days_late})


def leaky(df, seed):
    df = df.copy()
    df["high_risk"] = (df["days_late"] > 90).astype(int)
    X = df.drop(columns=["high_risk"])
    y = df["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def honest(df, seed):
    df = df.copy()
    df["high_risk"] = (df["days_late"] > 90).astype(int)
    X = df.drop(columns=["high_risk", "days_late"])
    y = df["high_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def run(seed=0):
    df = data(seed)
    return leaky(df, seed), honest(df, seed)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
