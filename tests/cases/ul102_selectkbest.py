import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)
X = rng.normal(size=(100, 10000))
y = rng.randint(0, 2, size=100)
X_new = SelectKBest(f_classif, k=20).fit_transform(X, y)  # expect: UL102
print(cross_val_score(LogisticRegression(), X_new, y, cv=5).mean())
