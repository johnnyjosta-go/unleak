"""UL103 - SMOTE before the train/test split.

A hard, imbalanced problem (3% positives). Oversampling first puts synthetic
neighbours of training rows into the test set.
"""

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RULE = "UL103"
TITLE = "SMOTE before the split (3% positives)"
METRIC = "F1 (positive class)"


def data(seed):
    return make_classification(n_samples=5000, n_features=20, n_informative=4, weights=[0.97],
                               class_sep=0.6, flip_y=0.02, random_state=seed)


def leaky(X, y, seed):
    X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=seed)
    model = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X_train, y_train)
    return f1_score(y_test, model.predict(X_test))


def honest(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    X_train, y_train = SMOTE(random_state=seed).fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X_train, y_train)
    return f1_score(y_test, model.predict(X_test))


def run(seed=0):
    X, y = data(seed)
    return leaky(X, y, seed), honest(X, y, seed)


if __name__ == "__main__":
    leak, fair = run()
    print(f"leaky  {METRIC}: {leak:.3f}\nhonest {METRIC}: {fair:.3f}")
