import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("sales.csv")
df["high_value"] = (df["revenue"] > 1000).astype(int)  # expect: UL303
y = df["high_value"]
X = df.drop(columns=["high_value", "customer_name"])
X_train, X_test, y_train, y_test = train_test_split(X, y)
RandomForestClassifier().fit(X_train, y_train)
