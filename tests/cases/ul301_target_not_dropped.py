import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("houses.csv")
y = df["price"]
X = df.drop(columns=["id"])  # expect: UL301
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(GradientBoostingRegressor().fit(X_train, y_train).score(X_test, y_test))
