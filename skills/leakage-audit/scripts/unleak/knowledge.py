"""What unleak knows about ML libraries.

Every name here is matched against the *last* component of a call
(``sklearn.preprocessing.StandardScaler`` -> ``StandardScaler``), so aliases
and ``module.Class`` spellings resolve to the same entry.
"""

from __future__ import annotations

import re

# --- estimator kinds -------------------------------------------------------

TRANSFORMER = "transformer"  # unsupervised, stateful preprocessing
ENCODER = "encoder"  # category -> code maps; leakage impact is negligible
SELECTOR = "selector"  # supervised feature selection / supervised projections
OVERSAMPLER = "oversampler"
UNDERSAMPLER = "undersampler"
TARGET_ENCODER = "target_encoder"
STAT = "stat"  # mean/std/quantile... computed with pandas or numpy
MODEL = "model"
SEARCH = "search"  # GridSearchCV & friends: a model that cross-validates
SPLITTER = "splitter"
STATELESS = "stateless"
UNKNOWN_TRANSFORMER = "unknown_transformer"

TRANSFORMERS = {
    # sklearn.preprocessing / impute
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
    "QuantileTransformer", "PowerTransformer", "KBinsDiscretizer",
    "SplineTransformer", "SimpleImputer", "KNNImputer", "IterativeImputer",
    "MissingIndicator", "VarianceThreshold",
    # decomposition / manifold / kernel approximation
    "PCA", "IncrementalPCA", "KernelPCA", "SparsePCA", "MiniBatchSparsePCA",
    "TruncatedSVD", "FastICA", "NMF", "MiniBatchNMF", "FactorAnalysis",
    "LatentDirichletAllocation", "DictionaryLearning",
    "MiniBatchDictionaryLearning", "Isomap", "LocallyLinearEmbedding",
    "SpectralEmbedding", "TSNE", "MDS", "UMAP", "Nystroem",
    # text
    "TfidfVectorizer", "CountVectorizer", "TfidfTransformer",
    # clustering used to build features
    "KMeans", "MiniBatchKMeans", "BisectingKMeans", "DBSCAN", "HDBSCAN",
    "OPTICS", "GaussianMixture", "BayesianGaussianMixture",
    "AgglomerativeClustering", "Birch", "MeanShift", "SpectralClustering",
    "AffinityPropagation",
    # keras preprocessing layers (fitted with .adapt) and the legacy Tokenizer
    "Normalization", "TextVectorization", "StringLookup", "IntegerLookup",
    "Discretization", "Tokenizer",
    # feature-engine / category_encoders (unsupervised but stateful)
    "MeanMedianImputer", "CategoricalImputer", "EndTailImputer",
    "RareLabelEncoder", "Winsorizer", "OutlierTrimmer",
    "EqualFrequencyDiscretiser", "EqualWidthDiscretiser",
    "CountFrequencyEncoder", "CountEncoder", "DropCorrelatedFeatures",
    "DropConstantFeatures", "DropDuplicateFeatures", "YeoJohnsonTransformer",
    "BoxCoxTransformer",
}

ENCODERS = {
    "LabelEncoder", "OrdinalEncoder", "OneHotEncoder", "LabelBinarizer",
    "MultiLabelBinarizer", "DictVectorizer", "BinaryEncoder", "BaseNEncoder",
    "HelmertEncoder", "SumEncoder", "BackwardDifferenceEncoder",
    "PolynomialEncoder",
}

STATELESS_TRANSFORMERS = {
    "Normalizer", "Binarizer", "FunctionTransformer", "HashingVectorizer",
    "PolynomialFeatures", "GaussianRandomProjection", "SparseRandomProjection",
    "RBFSampler", "AdditiveChi2Sampler", "SkewedChi2Sampler", "HashingEncoder",
    "ArbitraryNumberImputer",
}

SELECTORS = {
    "SelectKBest", "SelectPercentile", "SelectFpr", "SelectFdr", "SelectFwe",
    "GenericUnivariateSelect", "RFE", "RFECV", "SelectFromModel",
    "SequentialFeatureSelector", "ExhaustiveFeatureSelector", "BorutaPy",
    "DecisionTreeDiscretiser", "SelectBySingleFeaturePerformance",
    "SelectByShuffling", "RecursiveFeatureElimination",
    "RecursiveFeatureAddition", "SelectByTargetMeanPerformance",
    "SmartCorrelatedSelection", "MRMR",
    # supervised projections: fit_transform(X, y) uses the labels
    "LinearDiscriminantAnalysis", "NeighborhoodComponentsAnalysis",
    "PLSRegression", "PLSSVD", "PLSCanonical", "CCA",
}

TARGET_ENCODERS = {
    "TargetEncoder", "CatBoostEncoder", "WOEEncoder", "WoEEncoder",
    "JamesSteinEncoder", "MEstimateEncoder", "LeaveOneOutEncoder",
    "GLMMEncoder", "QuantileEncoder", "SummaryEncoder", "MeanEncoder",
}

OVERSAMPLERS = {
    "SMOTE", "SMOTENC", "SMOTEN", "ADASYN", "BorderlineSMOTE", "SVMSMOTE",
    "KMeansSMOTE", "RandomOverSampler", "SMOTETomek", "SMOTEENN",
}

UNDERSAMPLERS = {
    "RandomUnderSampler", "NearMiss", "TomekLinks", "EditedNearestNeighbours",
    "RepeatedEditedNearestNeighbours", "AllKNN", "CondensedNearestNeighbour",
    "OneSidedSelection", "NeighbourhoodCleaningRule",
    "InstanceHardnessThreshold", "ClusterCentroids",
}

SEARCHES = {
    "GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV",
    "HalvingRandomSearchCV", "BayesSearchCV", "TuneSearchCV",
    "TuneGridSearchCV", "OptunaSearchCV",
}

# splitter name -> (shuffles rows randomly by default, grouped, temporal)
SPLITTERS = {
    "KFold": (False, False, False),
    "StratifiedKFold": (False, False, False),
    "RepeatedKFold": (True, False, False),
    "RepeatedStratifiedKFold": (True, False, False),
    "ShuffleSplit": (True, False, False),
    "StratifiedShuffleSplit": (True, False, False),
    "LeaveOneOut": (False, False, False),
    "LeavePOut": (False, False, False),
    "GroupKFold": (False, True, False),
    "GroupShuffleSplit": (True, True, False),
    "StratifiedGroupKFold": (False, True, False),
    "LeaveOneGroupOut": (False, True, False),
    "LeavePGroupsOut": (False, True, False),
    "PredefinedSplit": (False, True, False),
    "TimeSeriesSplit": (False, False, True),
}

PIPELINE_BUILDERS = {
    "Pipeline", "make_pipeline", "ColumnTransformer", "make_column_transformer",
    "FeatureUnion", "make_union", "TransformedTargetRegressor",
}

# Model-ish class names we do not list one by one.
MODEL_NAME_RE = re.compile(
    r"(Classifier|Regressor|Regression|SVC|SVR|NB|Ridge|RidgeCV|Lasso|LassoCV|"
    r"ElasticNet|ElasticNetCV|Lars|LarsCV|LassoLars|Perceptron|"
    r"OrthogonalMatchingPursuit|ARDRegression|BayesianRidge|"
    r"QuadraticDiscriminantAnalysis|IsolationForest|OneClassSVM|"
    r"LocalOutlierFactor|EllipticEnvelope|Booster|Sequential|"
    r"CalibratedClassifierCV|IsotonicRegression)$"
)
MODEL_PREFIXES = (
    "XGB", "LGBM", "CatBoost", "RandomForest", "ExtraTrees", "GradientBoosting",
    "HistGradientBoosting", "AdaBoost", "Bagging", "Voting", "Stacking",
    "DecisionTree", "ExtraTree", "KNeighbors", "RadiusNeighbors", "MLP",
    "GaussianProcess", "SGD", "Logistic", "Linear", "TabNet", "Prophet",
    "NGBoost", "EBM", "ExplainableBoosting",
)
STATSMODELS_MODELS = {
    "OLS", "WLS", "GLS", "GLM", "Logit", "Probit", "MNLogit", "Poisson",
    "NegativeBinomial", "QuantReg", "ARIMA", "SARIMAX", "AutoReg", "VAR",
    "ExponentialSmoothing", "SimpleExpSmoothing", "Holt", "UnobservedComponents",
    "MixedLM", "ols", "logit", "glm",
}

# Constructors that build *data*, not estimators.
DATA_CONSTRUCTORS = {
    "DataFrame", "Series", "array", "asarray", "tensor", "Tensor", "from_numpy",
    "DMatrix", "QuantileDMatrix", "Dataset", "Pool", "TensorDataset",
    "DataLoader", "Subset", "matrix", "csr_matrix", "csc_matrix", "Index",
}

LOADERS = {
    "read_csv", "read_table", "read_parquet", "read_excel", "read_json",
    "read_feather", "read_pickle", "read_sql", "read_sql_query", "read_hdf",
    "read_orc", "read_stata", "read_sas", "read_spss", "read_fwf",
    "load", "loadtxt", "genfromtxt", "load_dataset", "load_from_disk",
    "ImageFolder", "open", "read_ipc", "scan_csv", "scan_parquet",
}
LOADER_PREFIXES = ("load_", "fetch_", "make_")

# --- functions -------------------------------------------------------------

SPLIT_FUNCS = {"train_test_split", "random_split", "temporal_train_test_split"}

CV_FUNCS = {
    "cross_val_score", "cross_validate", "cross_val_predict", "learning_curve",
    "validation_curve", "permutation_test_score",
}

METRIC_FUNCS = {
    "accuracy_score", "balanced_accuracy_score", "f1_score", "fbeta_score",
    "precision_score", "recall_score", "roc_auc_score",
    "average_precision_score", "log_loss", "brier_score_loss",
    "matthews_corrcoef", "cohen_kappa_score", "jaccard_score", "hamming_loss",
    "zero_one_loss", "top_k_accuracy_score", "classification_report",
    "confusion_matrix", "precision_recall_fscore_support", "roc_curve",
    "precision_recall_curve", "mean_squared_error", "root_mean_squared_error",
    "mean_absolute_error", "mean_absolute_percentage_error",
    "median_absolute_error", "r2_score", "explained_variance_score",
    "max_error", "mean_squared_log_error", "root_mean_squared_log_error",
    "mean_pinball_loss", "d2_absolute_error_score", "d2_tweedie_score",
    "mean_poisson_deviance", "mean_gamma_deviance", "mean_tweedie_deviance",
    "ndcg_score", "dcg_score", "det_curve", "auc",
}

SUPERVISED_STAT_FUNCS = {
    "mutual_info_classif", "mutual_info_regression", "chi2", "f_classif",
    "f_regression", "r_regression",
}

NUMPY_STAT_FUNCS = {
    "mean", "std", "var", "median", "min", "max", "amin", "amax", "ptp",
    "percentile", "quantile", "nanmean", "nanstd", "nanvar", "nanmedian",
    "nanmin", "nanmax", "nanpercentile", "nanquantile", "average",
}

# --- methods ---------------------------------------------------------------

FIT_METHODS = {
    "fit", "fit_transform", "fit_resample", "fit_sample", "fit_predict",
    "partial_fit", "adapt", "fit_on_texts", "fit_on_sequences",
}
# Methods whose result is data shaped by the estimator's fitted state.
APPLY_METHODS = {
    "transform", "inverse_transform", "predict", "predict_proba",
    "predict_log_proba", "decision_function", "score_samples", "kneighbors",
    "apply", "staged_predict", "staged_predict_proba", "predict_on_batch",
    "texts_to_sequences", "texts_to_matrix", "get_support",
    "get_feature_names_out", "__call__",
}
PREDICT_METHODS = {
    "predict", "predict_proba", "predict_log_proba", "decision_function",
    "staged_predict", "staged_predict_proba", "predict_on_batch",
}
SCORE_METHODS = {"score", "evaluate"}

PANDAS_STAT_METHODS = {
    "mean", "median", "std", "var", "min", "max", "quantile", "mode", "sem",
    "skew", "kurt", "kurtosis", "mad",
}
TIMESERIES_METHODS = {
    "shift", "rolling", "ewm", "expanding", "diff", "pct_change", "resample",
    "asfreq", "lag",
}
RANDOM_SUBSET_METHODS = {"sample"}

# Methods that keep a DataFrame's columns as they are.
COLUMN_PRESERVING_METHODS = {
    "copy", "dropna", "fillna", "reset_index", "sort_values", "sort_index",
    "sample", "query", "head", "tail", "astype", "replace", "to_numpy",
    "infer_objects", "convert_dtypes", "interpolate", "ffill", "bfill",
    "clip", "round", "abs", "drop_duplicates", "set_index", "nlargest",
    "nsmallest", "where", "mask", "rename_axis", "reindex", "truncate",
}
COLUMN_PRESERVING_ATTRS = {"values", "loc", "iloc"}
DATAFRAME_ATTRS = {
    "values", "loc", "iloc", "at", "iat", "shape", "columns", "index", "dtypes",
    "T", "size", "ndim", "empty", "str", "dt", "cat", "plot", "style", "attrs",
    "flags", "axes", "array", "name", "sparse",
}

# Methods that mutate their receiver (in addition to inplace=True).
MUTATING_METHODS = {
    "append", "extend", "insert", "update", "add", "setdefault", "appendleft",
}

SELECTION_FUNCS = {
    "max", "min", "argmax", "argmin", "nanargmax", "nanargmin", "sorted",
    "idxmax", "idxmin", "nlargest", "nsmallest", "sort_values", "argsort",
    "most_common",
}

EARLY_STOPPING_NAMES = {
    "EarlyStopping", "early_stopping", "ModelCheckpoint", "EarlyStoppingCallback",
}
EVAL_SET_KWARGS = {
    "eval_set", "validation_data", "valid_sets", "evals", "eval_X",
    "X_val", "valid_set",
}

# --- naming heuristics -----------------------------------------------------

TRAIN_TOKENS = {"train", "trn", "tr", "training", "trainset", "traindata"}
TEST_TOKENS = {"test", "tst", "holdout", "heldout", "oos", "te", "testset",
               "testdata", "hold"}
VAL_TOKENS = {"val", "valid", "validation", "dev", "vali", "va", "vl", "vld"}
TARGET_NAME_RE = re.compile(
    r"^(y|Y|y_?\w*|Y_?\w*|\w*_y|target|targets|label|labels|"
    r"\w*_target|\w*_labels?|target_\w*|labels?_\w*)$"
)
FEATURE_NAME_RE = re.compile(r"^(X|x|X_?\w*|x_\w*|\w*_X|features|feats|data_x)$")

ENTITY_COLUMN_RE = re.compile(
    r"^(patient|subject|user|customer|client|participant|person|member|player|"
    r"driver|household|family|account|device|session|study|animal|mouse|rat|"
    r"donor|speaker|author|document|doc|store|shop|site|hospital|school|"
    r"student|employee|company|firm|sensor|machine|vehicle|ship|well|image|"
    r"video|recording|trial|visit|encounter|admission|case)[_ ]?"
    r"(id|ids|no|num|number|code|key|uid|uuid|idx)$|^mrn$",
    re.IGNORECASE,
)


def last(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def estimator_kind(cls: str) -> str | None:
    """Kind of a class constructor, or None if it is not an estimator."""
    if cls in TRANSFORMERS:
        return TRANSFORMER
    if cls in ENCODERS:
        return ENCODER
    if cls in STATELESS_TRANSFORMERS:
        return STATELESS
    if cls in SELECTORS:
        return SELECTOR
    if cls in TARGET_ENCODERS:
        return TARGET_ENCODER
    if cls in OVERSAMPLERS:
        return OVERSAMPLER
    if cls in UNDERSAMPLERS:
        return UNDERSAMPLER
    if cls in SEARCHES:
        return SEARCH
    if cls in SPLITTERS:
        return SPLITTER
    if cls in STATSMODELS_MODELS:
        return MODEL
    if MODEL_NAME_RE.search(cls) or cls.startswith(MODEL_PREFIXES):
        return MODEL
    return None


def tokens(name: str) -> list[str]:
    """Split an identifier into lowercase tokens: XTrain_df2 -> [x, train, df, 2]."""
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Za-z])(\d)", r"\1_\2", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


# Names that mention a split but hold a statistic, a size or a setting:
# f_test, test_size, n_train, train_acc, test_scores ...
NOT_DATA_TOKENS = {
    "f", "t", "z", "chi", "chi2", "ks", "p", "pvalue", "pval", "size", "sizes",
    "score", "scores", "acc", "accuracy", "auc", "loss", "losses", "err",
    "error", "errors", "mse", "rmse", "mae", "r2", "metric", "metrics", "stat",
    "stats", "statistic", "n", "num", "len", "count", "ratio", "frac",
    "fraction", "percent", "pct", "time", "times", "path", "file", "files",
    "dir", "url", "name", "names", "mode", "flag", "result", "results",
    "report", "perf", "performance", "step", "steps", "epoch", "epochs",
    "batch", "batches", "fn", "func", "cfg", "config", "params", "kwargs",
    "max", "min", "mean", "avg", "std", "sum", "median", "var", "total", "best",
    "worst", "threshold", "thresh", "cutoff", "limit", "lim", "bound", "range",
    "lr", "alpha", "beta", "gamma", "eps", "tol", "seed", "label", "labels",
}


def split_role(name: str) -> str | None:
    """'train', 'test' or 'val' if a variable name says which split it holds."""
    toks = tokens(name.rsplit(".", 1)[-1])
    if any(t in NOT_DATA_TOKENS for t in toks):
        return None
    roles = set()
    for tok in toks:
        if tok in TRAIN_TOKENS:
            roles.add("train")
        elif tok in TEST_TOKENS:
            roles.add("test")
        elif tok in VAL_TOKENS:
            roles.add("val")
        else:
            # xtrain, dtrain, xtest, ytest, dfvalid ...
            for role, words in (("train", ("train",)), ("test", ("test",)),
                                ("val", ("valid",))):
                for w in words:
                    if tok.endswith(w) and 0 < len(tok) - len(w) <= 2:
                        roles.add(role)
                    elif tok.startswith(w) and tok[len(w):] in ("", "set", "data", "df", "x", "y", "idx", "ids", "index"):
                        roles.add(role)
    if len(roles) == 1:
        return roles.pop()
    return None


def path_role(path: str) -> str | None:
    """Split role from a file name like 'data/test.csv' or 'X_train.npy'."""
    base = re.split(r"[\\/]", path)[-1]
    base = base.split(".", 1)[0]
    return split_role(base)
