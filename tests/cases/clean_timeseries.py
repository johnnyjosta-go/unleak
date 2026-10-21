import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split

df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date")
df["lag_1"] = df["sales"].shift(1)
df["roll_7"] = df["sales"].shift(1).rolling(7).mean()
df["store_lag"] = df.groupby("store")["sales"].shift(7)
df["target"] = df["sales"].shift(-1)
df = df.dropna()
y = df["target"]
X = df.drop(columns=["target", "date"])
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
print(cross_val_score(RandomForestRegressor(), X, y, cv=TimeSeriesSplit(5)))
