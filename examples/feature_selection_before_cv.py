"""UL102 - feature selection before cross-validation.

Pure noise: 10,000 random features and coin-flip labels. Nothing can be
learned, so any accuracy above 50% is an artefact of the evaluation.
"""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

RULE = "UL102"
TITLE = "Feature selection before CV (pure noise data)"
METRIC = "CV accuracy"


def data(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(200, 10_000)), rng.integers(0, 2, size=200)


def leaky(X, y):
    X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
    return cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean()


def honest(X, y):
    model = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    return cross_val_score(model, X, y, cv=5).mean()


def run(seed=0):
    X, y = data(seed)
    return leaky(X, y), honest(X, y)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
