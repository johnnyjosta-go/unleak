import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

oof = np.zeros(len(X))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    sc = StandardScaler().fit(X_tr)
    model = LGBMClassifier().fit(sc.transform(X_tr), y[tr_idx])
    oof[va_idx] = model.predict_proba(sc.transform(X_va))[:, 1]
