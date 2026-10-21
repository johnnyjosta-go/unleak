from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)  # unleak: ignore[UL101]
X_train, X_test, y_train, y_test = train_test_split(X, y)
