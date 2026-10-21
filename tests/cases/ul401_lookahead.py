import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("prices.csv").sort_values("date")
df["ret_1"] = df["close"].pct_change()
df["next_ret"] = df["close"].pct_change().shift(-1)  # expect: UL401
df["smooth"] = df["close"].rolling(5, center=True).mean()  # expect: UL401
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
y = df["target"]
X = df.drop(columns=["target", "date"])
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
RandomForestClassifier().fit(X_train, y_train)
