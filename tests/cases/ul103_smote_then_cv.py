from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X, y)  # expect: UL103
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
print(cross_val_score(XGBClassifier(), X_bal, y_bal, cv=cv, scoring="f1").mean())
