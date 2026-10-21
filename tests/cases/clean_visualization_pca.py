import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_2d = PCA(n_components=2).fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = make_pipeline(StandardScaler(), SVC()).fit(X_train, y_train)
print(clf.score(X_test, y_test))
