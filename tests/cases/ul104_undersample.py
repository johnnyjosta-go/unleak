from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

X_u, y_u = RandomUnderSampler().fit_resample(X, y)  # expect: UL104
X_train, X_test, y_train, y_test = train_test_split(X_u, y_u)
