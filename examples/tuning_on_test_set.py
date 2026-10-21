"""UL204 - picking the best of 50 models by test score.

Data with *no* signal. The best test score among 50 tries is a winner's-curse
estimate; the chosen model scored on fresh rows shows the truth.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RULE = "UL204"
TITLE = "Best of 50 models chosen by test score (no real signal)"
METRIC = "accuracy"


def data(seed):
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(5000, 20)), rng.integers(0, 2, size=5000)
    return X[:600], y[:600], X[600:], y[600:]


def leaky(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    best_acc, best_model = 0.0, None
    for i in range(50):
        model = RandomForestClassifier(n_estimators=50, max_depth=1 + i % 8, random_state=i)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        if acc > best_acc:
            best_acc, best_model = acc, model
    return best_model, best_acc


def run(seed=0):
    X, y, X_new, y_new = data(seed)
    model, reported = leaky(X, y, seed)
    return reported, model.score(X_new, y_new)


if __name__ == "__main__":
    leak, fair = run()
    print(f"reported {METRIC}: {leak:.3f}\nreal     {METRIC}: {fair:.3f}")
