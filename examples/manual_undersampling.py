"""UL104 - balancing the classes by hand before splitting.

The pattern from many fraud-detection tutorials: keep all frauds, cut the
normal transactions down to the same count, then split. The test set is now
50% fraud, so precision describes a world that does not exist.
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

RULE = "UL104"
TITLE = "Manual undersampling before the split (0.5% fraud)"
METRIC = "precision (fraud)"


def data(seed):
    X, y = make_classification(n_samples=60_000, n_features=12, n_informative=6, weights=[0.995],
                               class_sep=1.0, flip_y=0.0, random_state=seed)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(12)])
    df["Class"] = y
    return df


def leaky(df, seed):
    df = df.sample(frac=1, random_state=seed)
    fraud = df.loc[df["Class"] == 1]
    normal = df.loc[df["Class"] == 0][: len(fraud)]
    balanced = pd.concat([fraud, normal])
    X = balanced.drop(columns=["Class"])
    y = balanced["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X_train, y_train)
    return precision_score(y_test, model.predict(X_test))


def honest(df, seed):
    train, test = train_test_split(df, test_size=0.3, stratify=df["Class"], random_state=seed)
    fraud = train.loc[train["Class"] == 1]
    normal = train.loc[train["Class"] == 0].sample(n=len(fraud), random_state=seed)
    balanced = pd.concat([fraud, normal])
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(balanced.drop(columns=["Class"]), balanced["Class"])
    return precision_score(test["Class"], model.predict(test.drop(columns=["Class"])))


def run(seed=0):
    df = data(seed)
    return leaky(df, seed), honest(df, seed)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
