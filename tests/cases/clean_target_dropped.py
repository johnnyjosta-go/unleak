import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("houses.csv")
target = "price"
y = df[target]
X = df.drop(columns=[target, "id"])
X_train, X_test, y_train, y_test = train_test_split(X, y)
LinearRegression().fit(X_train, y_train)

df2 = pd.read_csv("other.csv")
y2 = df2.pop("label")
X2 = df2
LinearRegression().fit(X2, y2)

features = [c for c in df.columns if c not in ("price", "id")]
LinearRegression().fit(df[features], df["price"])

cols = df.columns.drop(["price"])
LinearRegression().fit(df[cols], df["price"])
