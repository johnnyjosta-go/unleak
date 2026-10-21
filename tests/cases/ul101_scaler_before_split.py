import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

X = StandardScaler().fit_transform(X)  # expect: UL101
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
