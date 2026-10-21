from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
clf = RandomForestClassifier().fit(X_train, y_train)
print(clf.score(X_test, y_test))
pipe = make_pipeline(SMOTE(), RandomForestClassifier())
print(cross_val_score(pipe, X, y, cv=5))
