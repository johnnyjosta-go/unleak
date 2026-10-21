import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv("customers.csv")
df["city_churn_rate"] = df.groupby("city")["churn"].transform("mean")  # expect: UL302
df["roll"] = df["churn"].rolling(5).mean()  # expect: UL302
df["churn_lag"] = df.groupby("city")["churn"].shift(1)
y = df["churn"]
X = df.drop(columns=["churn", "city"])
print(cross_val_score(LogisticRegression(), X, y))
