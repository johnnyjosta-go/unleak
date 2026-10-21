---
name: leakage-audit
description: Find and fix data leakage in machine-learning code - preprocessing, feature selection or SMOTE fitted before the train/test split, target leakage and proxy features, look-ahead features and shuffled time-series splits, early stopping or model selection on the test set, entity or duplicate overlap between splits. Use when writing, reviewing or debugging training/evaluation code or notebooks (scikit-learn, pandas, XGBoost, LightGBM, CatBoost, Keras, PyTorch), before reporting any model metric, and whenever a result looks too good to be true.
license: MIT
---

# Leakage audit

A metric is only meaningful if neither the model nor any step that shaped its
inputs has seen the rows it is scored on. Leakage breaks that silently: the
code runs, the score is great, and it does not hold up on new data.

This skill bundles **unleak**, a static analyzer for `.py` and `.ipynb` files.
It needs only `python3` (3.10+), no install:

```bash
python3 <skill-dir>/scripts/run_unleak.py check <files-or-dirs>            # human-readable
python3 <skill-dir>/scripts/run_unleak.py check <paths> --format json      # for you to parse
python3 <skill-dir>/scripts/run_unleak.py explain UL103                    # why + fix for a rule
python3 <skill-dir>/scripts/run_unleak.py data train.csv test.csv \
        --target y --group customer_id --time date                         # checks on real data (needs pandas)
```

`<skill-dir>` is the directory containing this file. If `unleak` is installed
(`pip install unleak` or `uvx unleak`) the same commands work as `unleak check ...`.
Exit code 1 means findings at warning level or above.

## When writing ML code

Follow these rules by default, then run `check` on the files you wrote and fix
every finding before you run or report an evaluation.

1. **Split first.** Create the test set (or CV folds) before any fitted
   preprocessing, feature selection, resampling or target-derived feature.
2. **Fit inside a Pipeline.** Scalers, imputers, encoders, PCA, vectorizers and
   selectors go in `sklearn.pipeline.Pipeline`/`ColumnTransformer`; resamplers
   (SMOTE, undersampling) go in `imblearn.pipeline.Pipeline`. Then
   `cross_val_score`/`GridSearchCV` refit them inside every fold.
3. **Transform, never fit, held-out data.** `transform(X_test)`, not `fit_transform(X_test)`.
4. **Keep the target out of X.** Drop the target, anything computed from it,
   and any column recorded after the outcome (refund issued, discharge date...).
5. **Time series:** sort by time, split by time (`shuffle=False`, a date cutoff
   or `TimeSeriesSplit`), build lags with positive `shift()`, never `shift(-k)`
   or `center=True` for features.
6. **Grouped data:** when an entity (patient, user, store) has several rows,
   split with `GroupShuffleSplit`/`GroupKFold`.
7. **The test set is used once.** Tune hyperparameters, thresholds and early
   stopping on a validation split or with CV; score the final model on the
   test set one time and report that number.

Read [references/fixes.md](references/fixes.md) for copy-paste patterns.

## Auditing existing code

Use this workflow when asked to review, audit or debug ML code, or when a
score looks too good. Work through the phases in order and keep notes of each
finding with its evidence.

**1. Map the pipeline.** List every file involved, from raw data loading to the
reported metric: transformations, splits, CV, training, tuning, evaluation.
Note the unit of independence (row, patient, user, time period). If data is
read from a file another script or notebook wrote (`to_csv`, `np.save`,
`joblib.dump`), open that producer too; leaks often live upstream.

**2. Run the analyzer** on all of those files or their directory at once, so
cross-file flows are tracked:
`python3 <skill-dir>/scripts/run_unleak.py check <paths> --format json --min-severity info`.
Treat each finding as a strong hypothesis, not a verdict; `explain <RULE>`
gives the mechanism and the fix.

**3. Check what static analysis cannot see.** Go through
[references/taxonomy.md](references/taxonomy.md) against the code and the data
dictionary. Most often missed: proxy features recorded after the outcome,
duplicate or near-duplicate rows (augmentation before splitting, merged
sources), the same entity in train and test, aggregates computed over the
whole period, and repeated test-set peeking across notebook iterations.

**4. Check the data** when you can load it. Run `data` on the actual train and
test files with `--target`, `--group` and `--time` as applicable. It reports
identical rows across splits (UD001), shared entities (UD002), training rows
newer than test rows (UD003) and single columns that predict the target almost
perfectly (UD004).

**5. Verify every finding.** Confirm by fixing it in a copy and re-running the
evaluation: the change in the metric is the leak's size. Classify each finding:
- **confirmed**: the metric moves, or the mechanism is certain (e.g. target column in X)
- **plausible**: the mechanism is clear but you cannot run the code or data
- **rejected**: explain why it is safe, then add `# unleak: ignore[ULxxx]` with that reason on the line

**6. Fix and report.** Apply minimal fixes (usually a Pipeline or moving one
line below the split), re-run the analyzer until it is clean, re-run the
evaluation, and report using [references/report-template.md](references/report-template.md):
every finding, its status and evidence, the fix, and the metric before → after.

## Severity

- **error**: supervised information from evaluation rows reaches training
  (labels, test-set selection, duplicated rows). Expect materially inflated
  metrics; fix before anything else.
- **warning**: methodologically wrong; often small (plain scaling on large
  data) but can be large (imputers, PCA, TF-IDF or clustering on small data,
  class rebalancing). Fix it; state the measured effect.
- **info**: needs a data check to confirm (e.g. entity overlap).

## Rules

| Rule | Catches |
|---|---|
| UL101 | scaler / imputer / PCA / vectorizer / clusterer fitted on rows later used for evaluation |
| UL102 | supervised feature selection (SelectKBest, RFE, target correlation...) before the split |
| UL103 | SMOTE / oversampling before the split (including hand-rolled `resample` + `concat`) |
| UL104 | undersampling before the split (including cutting a class down by hand) |
| UL105 | target / WOE / mean encoder fitted before the split |
| UL106 | mean / median / std / min-max of the whole dataset used to fill or scale features |
| UL107 | preprocessing fitted on train and test concatenated |
| UL201 | `fit` / `fit_transform` on test data |
| UL202 | model trained on data that contains the rows it is evaluated on |
| UL203 | early stopping or best-iteration selection on the test set that is also reported |
| UL204 | loop picks the best model / hyperparameter by test-set score |
| UL301 | target column left in the feature matrix |
| UL302 | feature computed from the target (groupby target mean, unshifted rolling target) |
| UL303 | target derived from a column that is still a feature |
| UL401 | feature uses future values (`shift(-k)`, `rolling(center=True)`) |
| UL402 | random split / shuffled CV on data with lag or rolling features |
| UL403 | entity-id column present but split not grouped (info) |
| UD001-UD004 | data checks: duplicate rows, shared entities, temporal overlap, near-perfect single feature |

## Do not

- Do not report a metric from code with unresolved error-level findings
  without saying so explicitly.
- Do not silence a finding without a written reason in the ignore comment.
- Do not conclude "no leakage" only because the analyzer is clean; it cannot
  see proxy features, data provenance outside the code, or duplicates in the
  data. Say what you checked.
- Do not "fix" leakage by dropping the test evaluation or by evaluating on
  training data.
