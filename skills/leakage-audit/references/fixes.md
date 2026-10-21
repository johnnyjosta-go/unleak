# Leak-free patterns

Copy these shapes when writing or fixing ML code.

## Tabular: preprocessing + model in one Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
model = make_pipeline(pre, HistGradientBoostingClassifier())
print(cross_val_score(model, X_train, y_train, cv=5).mean())   # model selection on train only
model.fit(X_train, y_train)
print(model.score(X_test, y_test))                               # final, single test evaluation
```

## Imbalanced data: resample inside the training folds only

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline   # imblearn's Pipeline, not sklearn's

model = make_pipeline(StandardScaler(), SMOTE(random_state=0), RandomForestClassifier())
cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1")
model.fit(X_train, y_train)          # SMOTE never sees X_test
```

Often better: skip resampling, use `class_weight="balanced"` and pick the decision
threshold on a validation split. Evaluate on the untouched class distribution.

## Feature selection inside CV

```python
model = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression())
GridSearchCV(model, {"selectkbest__k": [10, 20, 50]}, cv=5).fit(X_train, y_train)
```

## Target encoding

```python
from sklearn.preprocessing import TargetEncoder   # cross-fits internally on fit_transform
model = make_pipeline(
    ColumnTransformer([("te", TargetEncoder(), high_card_cols)], remainder="passthrough"),
    HistGradientBoostingRegressor(),
)
model.fit(X_train, y_train)
```

By hand: compute category statistics on the training split only, map them onto the
test split, and use out-of-fold means for the training rows.

## Hyperparameters, early stopping, thresholds: validation data, not test data

```python
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=0)
model = XGBClassifier(n_estimators=5000, early_stopping_rounds=100)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
threshold = pick_threshold(y_val, model.predict_proba(X_val)[:, 1])
report(y_test, model.predict_proba(X_test)[:, 1] >= threshold)   # test used once
```

For an unbiased estimate of a whole tuning procedure, use nested CV:
`cross_val_score(GridSearchCV(pipe, grid, cv=3), X, y, cv=5)`.

## Time series

```python
df = df.sort_values("date")
df["lag_1"] = df.groupby("store")["sales"].shift(1)                     # past only
df["mean_7"] = df.groupby("store")["sales"].transform(lambda s: s.shift(1).rolling(7).mean())
df["target"] = df.groupby("store")["sales"].shift(-7)                   # future is fine for the TARGET
cutoff = "2025-01-01"
train, test = df[df.date < cutoff], df[df.date >= cutoff]
cv = TimeSeriesSplit(n_splits=5, gap=7)   # gap >= forecast horizon when targets overlap
```

## Grouped data

```python
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(X, y, groups=df["patient_id"]))
cross_val_score(model, X, y, groups=df["patient_id"], cv=GroupKFold(5))
```

## Deep learning

```python
norm = keras.layers.Normalization()
norm.adapt(X_train)                       # not the full dataset
vectorize = keras.layers.TextVectorization(max_tokens=20000)
vectorize.adapt(train_texts)
model.fit(X_train, y_train, validation_split=0.1, callbacks=[EarlyStopping(restore_best_weights=True)])
model.evaluate(X_test, y_test)            # once
```

PyTorch: compute normalisation statistics from the training `Subset` only, and
split by entity (patient/scan) before creating augmented crops.

## Deduplicate before splitting

```python
df = df.drop_duplicates(subset=feature_cols + [TARGET])
# augmentation / oversampling only after the split, only on the training part
```
