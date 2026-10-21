import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")
X = (X - X.mean(axis=0)) / X.std(axis=0)  # expect: UL106
X_train, X_test, y_train, y_test = train_test_split(X, y)
