"""UL203 - early stopping on the test set.

Gradient boosting on a small, noisy problem. The test set picks the number of
trees and then grades the result. Scoring the *same* model on rows it has never
touched shows how optimistic that grade is.
"""

from lightgbm import LGBMClassifier, early_stopping
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

RULE = "UL203"
TITLE = "Early stopping on the test set (small noisy data)"
METRIC = "ROC AUC"


def data(seed):
    X, y = make_classification(n_samples=3000, n_features=30, n_informative=3, flip_y=0.3,
                               class_sep=0.5, random_state=seed)
    # 600 rows for the experiment, the rest is an untouched "future" sample
    return X[:600], y[:600], X[600:], y[600:]


def leaky(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = LGBMClassifier(n_estimators=2000, learning_rate=0.05, verbose=-1, random_state=seed)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping(20, verbose=False)])
    return model, roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def run(seed=0):
    X, y, X_new, y_new = data(seed)
    model, reported = leaky(X, y, seed)
    return reported, roc_auc_score(y_new, model.predict_proba(X_new)[:, 1])


if __name__ == "__main__":
    leak, fair = run()
    print(f"reported {METRIC}: {leak:.3f}\nreal     {METRIC}: {fair:.3f}")
