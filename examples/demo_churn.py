"""A churn model with four leaks. Try: unleak check examples/demo_churn.py"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("churn.csv")
df["plan_churn_rate"] = df.groupby("plan")["churned"].transform("mean")
df["tenure"] = df["tenure"].fillna(df["tenure"].median())
X = df.drop(columns=["churned", "customer_id"])
y = df["churned"]
X, y = SMOTE(random_state=0).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = XGBClassifier(n_estimators=1000, early_stopping_rounds=30)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("F1:", f1_score(y_test, model.predict(X_test)))
