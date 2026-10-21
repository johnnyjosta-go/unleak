from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_pca = PCA(n_components=10).fit_transform(X)  # expect: UL101
scores = cross_val_score(SVC(), X_pca, y, cv=5)
print(scores.mean())
