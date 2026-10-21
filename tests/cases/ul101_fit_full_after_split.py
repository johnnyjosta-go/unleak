from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
scaler.fit(X)  # expect: UL101
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
clf = SVC().fit(X_train, y_train)
print(clf.score(X_test, y_test))
