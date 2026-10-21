import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date")
df["lag_1"] = df["sales"].shift(1)
df["roll_7"] = df["sales"].shift(1).rolling(7).mean()
df = df.dropna()
y = df["sales"]
X = df.drop(columns=["sales", "date"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # expect: UL402
RandomForestRegressor().fit(X_train, y_train)
