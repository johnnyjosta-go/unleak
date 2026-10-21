from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale(train, test):
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)


X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, X_test = scale(X_train, X_test)
