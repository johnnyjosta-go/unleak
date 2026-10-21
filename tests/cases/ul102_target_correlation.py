import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
corr = df.corr()["target"]  # expect: UL102
features = corr[abs(corr) > 0.1].index.drop("target")
X = df[features]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(LinearRegression().fit(X_train, y_train).score(X_test, y_test))
