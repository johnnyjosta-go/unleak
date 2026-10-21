# Leakage taxonomy and audit checklist

Kapoor & Narayanan ("Leakage and the reproducibility crisis in machine-learning-based
science", *Patterns*, 2023) found leakage reported in 17 scientific fields, affecting
294 papers, and grouped it into eight types. This checklist follows their grouping and
adds the evaluation mistakes that are common in applied and competition code. For each
item: what to look for, how to confirm, and the matching unleak rule when one exists.

## 1. No clean separation of training and evaluation data

**1a. Evaluating on training data.** Metrics computed on the rows the model was fitted
on, or a split that is created but never used. *Look for:* `model.score(X, y)` after
`model.fit(X, y)`; `cross_val_predict` predictions re-used as "test" predictions after
a refit. *Rule:* UL202 when a subset of the training data is used as test.

**1b. Preprocessing fitted on all rows.** Scaling, imputation, PCA, vectorizers,
clustering features, discretizers fitted before the split or outside the CV loop.
*Look for:* any `fit`/`fit_transform`/`adapt`/`fit_on_texts` above the split line;
pandas `fillna(df.mean())`, `(df - df.mean()) / df.std()`. *Rules:* UL101, UL106, UL107.

**1c. Feature selection on all rows.** Univariate tests, RFE, model-based
importance, mutual information or correlation with the target computed before the
split. On wide data this can produce high accuracy from pure noise. *Rule:* UL102.

**1d. Duplicates across splits.** The same record (or a near-copy) in train and test:
data augmentation before the split, oversampling before the split, datasets merged from
several sources, repeated measurements. *Confirm:* `unleak data` (UD001) for exact
copies; for near-duplicates compare nearest-neighbour distances test→train against
train→train. *Rules:* UL103, UD001.

## 2. Features that should not be available

**2a. The target in disguise.** The label column itself, a re-coding of it, or a column
it was computed from. *Look for:* `X = df.drop(columns=["id"])` with `y = df["label"]`;
labels created by thresholding or mapping another column. *Rules:* UL301, UL303, UD004.

**2b. Information recorded after the outcome.** Columns that exist only because the
outcome happened: `refund_issued` when predicting churn, `days_in_collections` for
default, `discharge_disposition` for readmission, `treatment_given` for diagnosis.
*Confirm:* for each top feature, ask "is this value known at prediction time?" Check
the data dictionary and how the column is populated. *Rule:* UD004 flags the extreme
cases; the rest needs judgement.

**2c. Target-derived features.** Group means of the target (hand-rolled target
encoding) computed on all rows, rolling statistics of the target that include the
current row, rank/percentile of the target. *Rules:* UL105, UL302.

**2d. Identifiers that encode the label.** IDs assigned in order of arrival, file
names or paths containing the class, image metadata from different scanners per class.
*Confirm:* single-feature AUC of ID-like columns (UD004); inspect sample file paths.

## 3. Dependence between training and test rows

**3a. Temporal leakage.** Training rows from after the test period; features computed
with future values (`shift(-k)`, centred windows, back-filling, full-period
aggregates, "as of today" snapshots joined to historical rows); overlapping prediction
windows split at random. *Rules:* UL401, UL402, UD003.

**3b. Non-independence (grouped data).** Several rows per patient, user, device,
store, session, or several crops/augmentations per image, split row-wise. The model
learns to recognise the entity. *Confirm:* `unleak data --group <id>` (UD002). *Rules:*
UL403, UD002.

**3c. Test distribution that does not match deployment.** Rebalanced or filtered test
sets (undersampling before the split, dropping hard cases), so precision and accuracy
describe an artificial population. *Rule:* UL104.

## 4. Re-using the test set

**4a. Early stopping, checkpoint selection or threshold tuning on the test set**
whose score is then reported. *Rule:* UL203.

**4b. Model or hyperparameter selection by test score**, including informal
selection: trying many variants in a notebook and keeping the one with the best test
number. *Rule:* UL204 catches loops; ask about the iteration history for the rest.

**4c. Feature engineering guided by test results** (adding features until the test
score improves). Not visible in code; ask, and if in doubt hold out a fresh test set.

## 5. Leaks outside the training script

- Preprocessing in another notebook that writes a cleaned file (unleak follows
  `to_csv`/`to_parquet`/`np.save`/`joblib.dump` → `read_*`/`np.load` across files it
  checks together).
- Feature stores or SQL views computed over the full history.
- Pre-trained embeddings or foundation models trained on data that includes the test
  set (benchmark contamination).
- Test sets assembled from data that was already used for training a previous model
  version.

## Quick confirmation recipes

```python
# Leak delta: fix one thing, re-run the evaluation, compare.
# Single-feature check: a feature with AUC >= 0.99 alone is suspicious.
from sklearn.metrics import roc_auc_score
for col in X_train.select_dtypes("number"):
    auc = roc_auc_score(y_train, X_train[col].fillna(X_train[col].median()))
    if max(auc, 1 - auc) > 0.95:
        print(col, round(max(auc, 1 - auc), 3))

# Entity overlap
shared = set(train["patient_id"]) & set(test["patient_id"])

# Exact duplicates across splits (feature columns only)
dup = test[features].merge(train[features].drop_duplicates(), how="inner")
```
