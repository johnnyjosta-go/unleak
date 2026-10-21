from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25)
best_k, best = None, 0
for k in range(1, 30):
    acc = accuracy_score(y_val, KNeighborsClassifier(k).fit(X_tr, y_tr).predict(X_val))
    if acc > best:
        best_k, best = k, acc
model = XGBClassifier(early_stopping_rounds=50)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
print(accuracy_score(y_test, model.predict(X_test)))
