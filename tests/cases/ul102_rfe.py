from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

selector = RFE(LogisticRegression(), n_features_to_select=5)
X_sel = selector.fit_transform(X, y)  # expect: UL102
X_train, X_test, y_train, y_test = train_test_split(X_sel, y)
