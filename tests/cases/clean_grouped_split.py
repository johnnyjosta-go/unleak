import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("visits.csv")
y = df["readmitted"]
X = df.drop(columns=["readmitted", "patient_id"])
gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
train_idx, test_idx = next(gss.split(X, y, groups=df["patient_id"]))
