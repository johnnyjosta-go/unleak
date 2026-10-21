import pandas as pd
from sklearn.tree import DecisionTreeClassifier

TARGET = "churned"
FEATURES = ["tenure", "monthly_charges", "churned", "contract"]
df = pd.read_csv("telco.csv")
X = df[FEATURES]  # expect: UL301
y = df[TARGET]
DecisionTreeClassifier().fit(X, y)
