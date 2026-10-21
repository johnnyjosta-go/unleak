import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {"lr": LogisticRegression(), "rf": RandomForestClassifier(), "svm": SVC()}
scores = []
for name, m in models.items():  # expect: UL204
    m.fit(X_train, y_train)
    scores.append(m.score(X_test, y_test))
best = list(models)[int(np.argmax(scores))]
