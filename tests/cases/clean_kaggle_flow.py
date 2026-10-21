import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop("Survived", axis=1)
y = train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
imp = SimpleImputer(strategy="median").fit(X_train)
clf = RandomForestClassifier().fit(imp.transform(X_train), y_train)
print(accuracy_score(y_val, clf.predict(imp.transform(X_val))))
final = RandomForestClassifier().fit(imp.transform(X), y)
submission = final.predict(imp.transform(test))
