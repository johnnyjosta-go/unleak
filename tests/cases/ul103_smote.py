from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)  # expect: UL103
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3)
clf = RandomForestClassifier().fit(X_train, y_train)
print(f1_score(y_test, clf.predict(X_test)))
