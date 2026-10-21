"""UL101 - scaling before the split.

The classic textbook leak, measured honestly: with plenty of data and a plain
StandardScaler the inflation is usually tiny. unleak rates it a warning, not
an error, for that reason.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RULE = "UL101"
TITLE = "StandardScaler before the split (breast cancer data)"
METRIC = "accuracy"


def data(seed):
    return load_breast_cancer(return_X_y=True)


def leaky(X, y, seed):
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    return LogisticRegression(max_iter=5000).fit(X_train, y_train).score(X_test, y_test)


def honest(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_train, y_train)
    return model.score(X_test, y_test)


def run(seed=0):
    X, y = data(seed)
    return leaky(X, y, seed), honest(X, y, seed)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
