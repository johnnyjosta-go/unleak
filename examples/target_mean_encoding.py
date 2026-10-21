"""UL302 - hand-rolled target encoding on the full dataset.

A high-cardinality category with no relation to the label. Encoding each
category by its mean label *before* splitting copies the test labels into the
test features.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RULE = "UL302"
TITLE = "Target mean encoding before the split (no real signal)"
METRIC = "ROC AUC"


def data(seed):
    rng = np.random.default_rng(seed)
    n = 4000
    return pd.DataFrame({
        "zip_code": rng.integers(0, 1500, n).astype(str),
        "income": rng.normal(50, 10, n),
        "default": rng.integers(0, 2, n),
    })


def leaky(df, seed):
    df = df.copy()
    df["zip_rate"] = df.groupby("zip_code")["default"].transform("mean")
    X = df.drop(columns=["default", "zip_code"])
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = LogisticRegression().fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def honest(df, seed):
    train, test = train_test_split(df, test_size=0.25, random_state=seed)
    rates = train.groupby("zip_code")["default"].mean()
    prior = train["default"].mean()
    train = train.assign(zip_rate=train["zip_code"].map(rates))
    test = test.assign(zip_rate=test["zip_code"].map(rates).fillna(prior))
    cols = ["income", "zip_rate"]
    model = LogisticRegression().fit(train[cols], train["default"])
    return roc_auc_score(test["default"], model.predict_proba(test[cols])[:, 1])


def run(seed=0):
    df = data(seed)
    return leaky(df, seed), honest(df, seed)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
