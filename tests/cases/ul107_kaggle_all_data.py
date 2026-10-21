import pandas as pd
from sklearn.linear_model import Ridge

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["SalePrice"]
all_data = pd.concat([train.drop(columns=["SalePrice"]), test])
all_data = all_data.fillna(all_data.median())  # expect: UL107
X = all_data[: len(train)]
X_test = all_data[len(train):]
model = Ridge().fit(X, y)
preds = model.predict(X_test)
