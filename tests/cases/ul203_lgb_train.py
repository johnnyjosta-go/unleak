import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
dtrain = lgb.Dataset(X_train, y_train)
dtest = lgb.Dataset(X_test, y_test)
booster = lgb.train(params, dtrain, valid_sets=[dtest], callbacks=[lgb.early_stopping(100)])  # expect: UL203
print(roc_auc_score(y_test, booster.predict(X_test)))
