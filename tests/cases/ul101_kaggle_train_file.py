import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
X = train.drop(columns=["Survived"])
y = train["Survived"]
imp = SimpleImputer(strategy="median").fit(X)  # expect: UL101
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier().fit(imp.transform(X_tr), y_tr)
print(clf.score(imp.transform(X_val), y_val))
