from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

best_k, best_acc = None, 0.0
for k in range(1, 30):  # expect: UL204
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    if acc > best_acc:
        best_k, best_acc = k, acc
print(best_k, best_acc)
