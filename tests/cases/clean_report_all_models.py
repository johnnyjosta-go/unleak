from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

for name, m in {"lr": LogisticRegression(), "rf": RandomForestClassifier()}.items():
    m.fit(X_train, y_train)
    print(name, m.score(X_test, y_test))
