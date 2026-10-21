"""Rule catalogue: ids, severities and the text shown by `unleak explain`."""

from __future__ import annotations

from dataclasses import dataclass

ERROR, WARNING, INFO = "error", "warning", "info"
SEVERITY_ORDER = {INFO: 0, WARNING: 1, ERROR: 2}


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    severity: str
    title: str
    why: str
    fix: str
    bad: str
    good: str


_RULES = [
    Rule(
        "UL101", "preprocessing-before-split", WARNING,
        "Preprocessing fitted on rows that end up in the test set",
        "A scaler, imputer, PCA, vectorizer, clusterer or similar was fitted on "
        "data that includes held-out rows. Their statistics (means, variances, "
        "vocabulary, components) leak into the training features, so the test "
        "score is no longer an estimate of performance on unseen data. The "
        "effect is often small for plain scaling and can be large for "
        "imputers, PCA, TF-IDF and clustering on small datasets.",
        "Split first, then fit preprocessing on the training rows only. The "
        "simplest way is a Pipeline: cross-validation and .fit() then refit "
        "the step inside every training fold.",
        "X = StandardScaler().fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)",
        "model = make_pipeline(StandardScaler(), LogisticRegression())\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "model.fit(X_train, y_train)",
    ),
    Rule(
        "UL102", "feature-selection-before-split", ERROR,
        "Supervised feature selection saw the test labels",
        "Feature selection that looks at the target (SelectKBest, RFE, "
        "SelectFromModel, mutual information, correlation with the target, "
        "LDA...) was run on rows that are later used for testing. The "
        "selected features are partly chosen *because* they fit the test "
        "labels. On high-dimensional data this alone can turn pure noise into "
        "an apparently accurate model.",
        "Do selection inside the training data only, e.g. "
        "make_pipeline(SelectKBest(k=20), model) evaluated with "
        "cross_val_score, or fit the selector on X_train.",
        "X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)\n"
        "cross_val_score(LogisticRegression(), X_sel, y)",
        "pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())\n"
        "cross_val_score(pipe, X, y)",
    ),
    Rule(
        "UL103", "oversampling-before-split", ERROR,
        "Oversampling ran before the split",
        "SMOTE, ADASYN and RandomOverSampler create synthetic or duplicated "
        "minority rows. Done before splitting, near-copies of the same "
        "original rows land on both sides of the split, so the model is "
        "tested on rows it has effectively already seen. Minority-class "
        "recall and F1 are typically inflated dramatically.",
        "Resample only the training data. With cross-validation, use "
        "imblearn.pipeline.make_pipeline(SMOTE(), model) so resampling "
        "happens inside each training fold. Never resample the test set.",
        "X_res, y_res = SMOTE().fit_resample(X, y)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)\n"
        "X_train, y_train = SMOTE().fit_resample(X_train, y_train)",
    ),
    Rule(
        "UL104", "undersampling-before-split", WARNING,
        "Undersampling ran before the split",
        "Undersampling before the split changes the class balance of the test "
        "set, so test metrics describe an artificial, balanced population "
        "instead of the real one. Cleaning methods (Tomek links, ENN) also "
        "use the test rows to decide what to drop.",
        "Split first and resample only the training data (or use an "
        "imblearn Pipeline). Evaluate on the untouched class distribution.",
        "X_res, y_res = RandomUnderSampler().fit_resample(X, y)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)\n"
        "X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)",
    ),
    Rule(
        "UL105", "target-encoding-before-split", ERROR,
        "Target encoding learned from the test labels",
        "A target/mean/WOE encoder replaces each category with a statistic of "
        "the target. Fitted before the split, test rows' own labels are baked "
        "into their encoded features - a direct label leak that is strongest "
        "for rare categories.",
        "Fit the encoder on training rows only (inside a Pipeline or the CV "
        "loop). sklearn's TargetEncoder cross-fits inside fit_transform, but "
        "only when fitted on the training split.",
        "X['city'] = TargetEncoder().fit_transform(X[['city']], y)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)",
        "pipe = make_pipeline(TargetEncoder(), HistGradientBoostingClassifier())\n"
        "pipe.fit(X_train, y_train)",
    ),
    Rule(
        "UL106", "global-statistics-before-split", WARNING,
        "Features filled or scaled with statistics of the whole dataset",
        "A mean, median, std, min/max or quantile computed on all rows was "
        "used to impute or normalise features before the train/test split. "
        "It is the hand-written version of UL101: test rows influence the "
        "values seen during training.",
        "Compute the statistic on the training split and reuse it for the "
        "test split, or use SimpleImputer/StandardScaler inside a Pipeline.",
        "df['age'] = df['age'].fillna(df['age'].mean())\n"
        "train, test = train_test_split(df)",
        "train, test = train_test_split(df)\n"
        "mean_age = train['age'].mean()\n"
        "train['age'] = train['age'].fillna(mean_age)\n"
        "test['age'] = test['age'].fillna(mean_age)",
    ),
    Rule(
        "UL107", "fit-on-train-and-test", WARNING,
        "Preprocessing fitted on train and test combined",
        "The training and test sets were concatenated and a transformer was "
        "fitted on the result. This is common in competitions where test "
        "labels are unknown, but any score measured on held-out rows that "
        "went into the fit (including a validation split carved from train) "
        "is optimistic.",
        "Fit preprocessing on the training rows only, or accept the "
        "transductive setup knowingly and do not report its validation "
        "score as an unbiased estimate.",
        "full = pd.concat([train, test])\n"
        "scaler.fit(full[cols])",
        "scaler.fit(train[cols])\n"
        "test[cols] = scaler.transform(test[cols])",
    ),
    Rule(
        "UL201", "fit-on-test-data", ERROR,
        "A model or transformer was fitted on the test set",
        "Calling fit / fit_transform / fit_resample on test data either "
        "trains on the rows you evaluate on, or (for preprocessing) "
        "re-learns the transformation from the test data so train and test "
        "are encoded inconsistently. Typical bug: "
        "X_test = scaler.fit_transform(X_test).",
        "Fit on the training split, then call only transform / predict on "
        "the test split.",
        "X_train = scaler.fit_transform(X_train)\n"
        "X_test = scaler.fit_transform(X_test)",
        "X_train = scaler.fit_transform(X_train)\n"
        "X_test = scaler.transform(X_test)",
    ),
    Rule(
        "UL202", "model-trained-on-test-rows", ERROR,
        "The model was trained on data that contains the test rows",
        "The model (or search) was fitted on the full dataset and then "
        "evaluated on a subset of that same dataset. Every test row was "
        "part of training, so the score measures memorisation, not "
        "generalisation.",
        "Fit on X_train only. If you refit on all data for deployment, do it "
        "after the evaluation and do not evaluate that refitted model on "
        "the old test split.",
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "model.fit(X, y)\n"
        "model.score(X_test, y_test)",
        "model.fit(X_train, y_train)\n"
        "model.score(X_test, y_test)",
    ),
    Rule(
        "UL203", "early-stopping-on-test", ERROR,
        "Early stopping / best-model selection used the test set",
        "The test set was passed as eval_set / validation_data while early "
        "stopping or best-iteration selection was active, and the same test "
        "set is used to report the final metric. The number of trees or "
        "epochs was tuned on the test data, so the reported score is "
        "optimistic.",
        "Carve a separate validation split from the training data for early "
        "stopping and keep the test set for the final, single evaluation.",
        "model.fit(X_train, y_train, eval_set=[(X_test, y_test)],\n"
        "          early_stopping_rounds=50)\n"
        "accuracy_score(y_test, model.predict(X_test))",
        "X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)\n"
        "model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])\n"
        "accuracy_score(y_test, model.predict(X_test))",
    ),
    Rule(
        "UL204", "tuning-on-test-set", WARNING,
        "Hyperparameters or models were selected by test-set score",
        "A loop fits several models or hyperparameter values, scores each on "
        "the test set and keeps the best. The winning score is then a "
        "maximum over noisy test estimates - it is biased upward, and the "
        "test set has become a validation set.",
        "Select with cross-validation (GridSearchCV) or a separate "
        "validation split, then evaluate the single chosen model once on "
        "the test set.",
        "for k in range(1, 30):\n"
        "    knn = KNeighborsClassifier(k).fit(X_train, y_train)\n"
        "    acc = knn.score(X_test, y_test)\n"
        "    if acc > best: best, best_k = acc, k",
        "search = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 30)})\n"
        "search.fit(X_train, y_train)\n"
        "search.score(X_test, y_test)",
    ),
    Rule(
        "UL301", "target-in-features", ERROR,
        "The target column is still in the feature matrix",
        "The column used as y was not removed from X, so the model can read "
        "the answer directly. Scores near 100% are the usual symptom.",
        "Drop the target (and anything derived from it) when building X, "
        "e.g. X = df.drop(columns=[target]).",
        "y = df['price']\n"
        "X = df.drop(columns=['id'])",
        "y = df['price']\n"
        "X = df.drop(columns=['id', 'price'])",
    ),
    Rule(
        "UL302", "feature-built-from-target", ERROR,
        "A feature was computed from the target",
        "A feature column was derived from the target column on data that "
        "includes held-out rows - for example a groupby mean of the target "
        "(hand-rolled target encoding) or a rolling window that includes the "
        "current target. The label of each row leaks into its own features.",
        "Compute target-based features on the training split only, using "
        "out-of-fold estimates, or shift time-series windows so they only "
        "use past targets (.shift(1) before .rolling()).",
        "df['city_rate'] = df.groupby('city')['churn'].transform('mean')\n"
        "X = df.drop(columns=['churn'])",
        "rates = train.groupby('city')['churn'].mean()\n"
        "test['city_rate'] = test['city'].map(rates)",
    ),
    Rule(
        "UL303", "target-proxy-in-features", ERROR,
        "The target was derived from a column that is still a feature",
        "The label was computed from another column (a threshold, mapping or "
        "formula), and that source column was left in X. The model can "
        "reconstruct the label from it without learning anything useful.",
        "Drop the column(s) the target was derived from, or redefine the "
        "task so the features are only things known before the outcome.",
        "df['high_value'] = (df['revenue'] > 1000).astype(int)\n"
        "X = df.drop(columns=['high_value'])",
        "df['high_value'] = (df['revenue'] > 1000).astype(int)\n"
        "X = df.drop(columns=['high_value', 'revenue'])",
    ),
    Rule(
        "UL401", "lookahead-feature", ERROR,
        "A feature uses values from the future",
        "A feature was built with a negative shift or a centred rolling "
        "window, so each row can see values recorded after it. Fine for "
        "constructing the *target*, never for a feature. (If your data is "
        "sorted newest-first, a negative shift points to the past; add an "
        "ignore comment in that case.)",
        "Use positive shifts for lag features and center=False (the default) "
        "for rolling windows.",
        "df['next_close'] = df['close'].shift(-1)\n"
        "df['trend'] = df['close'].rolling(7, center=True).mean()",
        "df['prev_close'] = df['close'].shift(1)\n"
        "df['trend'] = df['close'].rolling(7).mean().shift(1)",
    ),
    Rule(
        "UL402", "shuffled-time-series-split", ERROR,
        "Time-series features with a random split",
        "The data has lag/rolling/diff features, but rows were split "
        "randomly (train_test_split shuffles by default; ShuffleSplit and "
        "KFold(shuffle=True) too). Training rows then come from after the "
        "test rows, and overlapping windows copy test information into "
        "training features.",
        "Split by time: train_test_split(..., shuffle=False) on data sorted "
        "by time, TimeSeriesSplit for cross-validation, or a date cutoff.",
        "df['lag_1'] = df['sales'].shift(1)\n"
        "train_test_split(X, y, test_size=0.2)",
        "df = df.sort_values('date')\n"
        "train_test_split(X, y, test_size=0.2, shuffle=False)\n"
        "cross_val_score(model, X, y, cv=TimeSeriesSplit(5))",
    ),
    Rule(
        "UL403", "ungrouped-entity-split", INFO,
        "Rows belong to entities but the split is not grouped",
        "The data has an entity identifier (patient_id, user_id...) but the "
        "split is row-wise. If an entity has several rows, the same patient "
        "or user ends up in both train and test and the model is rewarded "
        "for recognising them. Verify with `unleak data --group <col>`.",
        "Split with GroupShuffleSplit / GroupKFold / StratifiedGroupKFold "
        "using the entity id as groups.",
        "train_test_split(X, y)  # df has several rows per patient_id",
        "GroupShuffleSplit(test_size=0.2).split(X, y, groups=df['patient_id'])",
    ),
    # --- data checks (unleak data) -----------------------------------------
    Rule(
        "UD001", "duplicate-rows-across-splits", ERROR,
        "Identical rows appear in both train and test",
        "Rows with identical feature values are present on both sides of "
        "the split. The model is tested on examples it was trained on - "
        "common after oversampling, augmentation or merging data sources.",
        "Deduplicate before splitting, or split by a key that keeps "
        "duplicates together.",
        "df = pd.concat([df, df_augmented]); train_test_split(df)",
        "df = df.drop_duplicates(subset=feature_cols); train_test_split(df)",
    ),
    Rule(
        "UD002", "group-overlap", ERROR,
        "The same entity appears in train and test",
        "Values of the group column (e.g. patient_id) occur in both splits, "
        "so the test set is not made of new entities.",
        "Use GroupShuffleSplit / GroupKFold with this column as groups.",
        "train_test_split(df)",
        "GroupShuffleSplit().split(df, groups=df['patient_id'])",
    ),
    Rule(
        "UD003", "temporal-overlap", WARNING,
        "Training rows are newer than test rows",
        "Some training rows have timestamps later than the earliest test "
        "row. For forecasting-type problems this means the model learns "
        "from the future of the period it is evaluated on.",
        "Use a time cutoff: train on rows before date T, test on rows after.",
        "train_test_split(df)",
        "train, test = df[df.date < cutoff], df[df.date >= cutoff]",
    ),
    Rule(
        "UD004", "near-perfect-feature", WARNING,
        "A single feature predicts the target almost perfectly",
        "One column alone separates the classes (AUC >= 0.99) or tracks the "
        "target (|correlation| >= 0.99). Real-world signal is rarely that "
        "clean; this is the classic signature of target leakage through a "
        "proxy column (an outcome recorded after the fact, an id assigned by "
        "label, a duplicated target).",
        "Check how the column is produced and whether it is available at "
        "prediction time. Drop it if not.",
        "X = df.drop(columns=['default'])  # still has 'days_in_collections'",
        "X = df.drop(columns=['default', 'days_in_collections'])",
    ),
]

RULES: dict[str, Rule] = {r.id: r for r in _RULES}


def get(rule_id: str) -> Rule:
    return RULES[rule_id.upper()]


def matches(rule_id: str, patterns: list[str]) -> bool:
    """True if rule_id matches any of the given ids or prefixes (UL1, UL101)."""
    return any(rule_id.startswith(p.strip().upper()) for p in patterns if p.strip())
