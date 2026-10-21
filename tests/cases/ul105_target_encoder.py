import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

df = pd.read_csv("loans.csv")
y = df["default"]
X = df.drop(columns=["default"])
X["city"] = TargetEncoder().fit_transform(X[["city"]], y)  # expect: UL105
X_train, X_test, y_train, y_test = train_test_split(X, y)
