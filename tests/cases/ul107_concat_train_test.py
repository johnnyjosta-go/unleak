import pandas as pd
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
cols = ["a", "b", "c"]
full = pd.concat([train, test])
scaler = StandardScaler().fit(full[cols])  # expect: UL107
train[cols] = scaler.transform(train[cols])
test[cols] = scaler.transform(test[cols])
