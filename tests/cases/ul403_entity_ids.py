import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("visits.csv")
y = df["readmitted"]
X = df.drop(columns=["readmitted", "patient_id"])
X_train, X_test, y_train, y_test = train_test_split(X, y)  # expect: UL403
