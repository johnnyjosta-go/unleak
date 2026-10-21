# What each leak does to the numbers

Mean over 5 random seeds. Every leaky pipeline below is flagged by `unleak check examples/`;
every honest one passes. Reproduce with `python examples/run_all.py`.

| Example | Rule | Metric | Leaky pipeline reports | Honest estimate |
|---|---|---|---:|---:|
| [Feature selection before CV (pure noise data)](feature_selection_before_cv.py) | UL102 | CV accuracy | **0.818** | 0.514 |
| [SMOTE before the split (3% positives)](smote_before_split.py) | UL103 | F1 (positive class) | **0.978** | 0.207 |
| [Manual undersampling before the split (0.5% fraud)](manual_undersampling.py) | UL104 | precision (fraud) | **0.909** | 0.056 |
| [Target mean encoding before the split (no real signal)](target_mean_encoding.py) | UL302 | ROC AUC | **0.832** | 0.496 |
| [Label defined from a column that stays a feature](target_proxy.py) | UL303 | accuracy | **1.000** | 0.863 |
| [Look-ahead feature (random-walk prices)](lookahead_feature.py) | UL401 | direction accuracy | **0.727** | 0.490 |
| [Shuffled split on a time series (random walk)](shuffled_time_series_split.py) | UL402 | accuracy (10-day direction) | **0.647** | 0.514 |
| [Early stopping on the test set (small noisy data)](early_stopping_on_test.py) | UL203 | ROC AUC | **0.697** | 0.679 |
| [Best of 50 models chosen by test score (no real signal)](tuning_on_test_set.py) | UL204 | accuracy | **0.556** | 0.498 |
| [StandardScaler before the split (breast cancer data)](scaler_before_split.py) | UL101 | accuracy | **0.973** | 0.972 |

For early stopping and model selection on the test set, the honest estimate is the *same chosen model* scored on fresh rows that played no part in training or selection.

Environment: Python 3.13.16, scikit-learn 1.9.1.
