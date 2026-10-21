import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("creditcard.csv").sample(frac=1)
fraud = df.loc[df["Class"] == 1]
normal = df.loc[df["Class"] == 0][:492]
balanced = pd.concat([fraud, normal])  # expect: UL104
X = balanced.drop("Class", axis=1)
y = balanced["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))
