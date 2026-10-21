from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
search = GridSearchCV(SVC(), {"C": [0.1, 1, 10]}, cv=5)
search.fit(X, y)  # expect: UL202
print(search.score(X_test, y_test))
