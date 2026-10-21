import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv("churn.csv")
df_majority = df[df["target"] == 0]
df_minority = df[df["target"] == 1]
df_minority_up = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
df_up = pd.concat([df_majority, df_minority_up])  # expect: UL103
y = df_up["target"]
X = df_up.drop("target", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
