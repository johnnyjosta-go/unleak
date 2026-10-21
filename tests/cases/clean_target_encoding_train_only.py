import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("customers.csv")
train, test = train_test_split(df, test_size=0.2)
rates = train.groupby("city")["churn"].mean()
train["city_rate"] = train["city"].map(rates)
test["city_rate"] = test["city"].map(rates)
