"""Data-level leakage checks on an actual train/test pair (needs pandas)."""

from __future__ import annotations

import os

from . import rules
from .checker import Finding

AUC_LIMIT = 0.99
CORR_LIMIT = 0.99
MAX_CATEGORIES = 50


def _require_pandas():
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise SystemExit("`unleak data` needs pandas: pip install 'unleak[data]'") from exc
    return pd


def load_table(path: str):
    pd = _require_pandas()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext in (".json", ".jsonl", ".ndjson"):
        return pd.read_json(path, lines=ext != ".json")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext in (".tsv", ".tab"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _finding(rule_id: str, path: str, message: str) -> Finding:
    r = rules.get(rule_id)
    return Finding(rule=r.id, name=r.name, severity=r.severity, path=path, line=0, column=0,
                   end_line=0, message=message, title=r.title)


def _rank_auc(x, y) -> float | None:
    """ROC AUC of a single numeric column for a binary target (Mann-Whitney)."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    pos = y == y.max()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = x.rank(method="average")
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(auc, 1 - auc)


def check(train_path: str, test_path: str, target: str | None = None, group: str | None = None,
          time: str | None = None, ignore: list[str] | None = None, train=None, test=None) -> list[Finding]:
    pd = _require_pandas()
    train = load_table(train_path) if train is None else train
    test = load_table(test_path) if test is None else test
    ignore = set(ignore or [])
    label = f"{os.path.basename(train_path)} vs {os.path.basename(test_path)}"
    out: list[Finding] = []

    shared = [c for c in train.columns if c in test.columns and c not in ignore]
    features = [c for c in shared if c not in {target, group, time}]

    # UD001: identical feature rows on both sides
    if features:
        a, b = train[features].copy(), test[features].copy()
        for c in features:  # CSV vs Parquet: compare mismatched dtypes as text
            if a[c].dtype != b[c].dtype:
                a[c], b[c] = a[c].astype(str), b[c].astype(str)
        h_train = set(pd.util.hash_pandas_object(a, index=False).tolist())
        h_test = pd.util.hash_pandas_object(b, index=False)
        dup = h_test.isin(h_train)
        n = int(dup.sum())
        if n:
            rows = ", ".join(str(i) for i in list(test.index[dup.values])[:5])
            out.append(_finding("UD001", label,
                                f"{n} of {len(test)} test rows ({n / len(test):.1%}) have the exact same feature "
                                f"values as a training row (e.g. test rows {rows})"))

    # UD002: same entity on both sides
    if group:
        for name, df in (("train", train), ("test", test)):
            if group not in df.columns:
                raise SystemExit(f"--group column '{group}' not found in the {name} file")
        overlap = set(train[group].dropna().unique()) & set(test[group].dropna().unique())
        if overlap:
            rows = int(test[group].isin(overlap).sum())
            sample = ", ".join(str(v) for v in sorted(overlap, key=str)[:3])
            out.append(_finding("UD002", label,
                                f"{len(overlap)} '{group}' values (e.g. {sample}) appear in both files; "
                                f"{rows} test rows ({rows / len(test):.1%}) belong to entities seen in training"))

    # UD003: training rows newer than the test period
    if time:
        for name, df in (("train", train), ("test", test)):
            if time not in df.columns:
                raise SystemExit(f"--time column '{time}' not found in the {name} file")
        t_train = pd.to_datetime(train[time], errors="coerce")
        t_test = pd.to_datetime(test[time], errors="coerce")
        start = t_test.min()
        if pd.notna(start):
            later = int((t_train > start).sum())
            if later:
                out.append(_finding("UD003", label,
                                    f"{later} training rows ({later / len(train):.1%}) are dated after the first "
                                    f"test row ({start}); the model can learn from the test period"))

    # UD004: a single column that (almost) is the target
    if target:
        if target not in train.columns:
            raise SystemExit(f"--target column '{target}' not found in {os.path.basename(train_path)}")
        y = train[target]
        numeric_target = pd.api.types.is_numeric_dtype(y) and y.nunique() > 2
        binary = y.nunique() == 2
        for col in features:
            x = train[col]
            if x.nunique(dropna=True) <= 1:
                continue
            if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
                if binary:
                    auc = _rank_auc(x.astype(float), (y == y.dropna().unique()[0]).astype(int))
                    if auc is not None and auc >= AUC_LIMIT:
                        out.append(_finding("UD004", os.path.basename(train_path),
                                            f"column '{col}' alone separates '{target}' with AUC {auc:.3f}"))
                elif numeric_target:
                    corr = x.rank().corr(y.astype(float).rank())  # Spearman without scipy
                    if pd.notna(corr) and abs(corr) >= CORR_LIMIT:
                        out.append(_finding("UD004", os.path.basename(train_path),
                                            f"column '{col}' has rank correlation {corr:+.3f} with '{target}'"))
            elif x.nunique() <= MAX_CATEGORIES and not numeric_target:
                frame = pd.DataFrame({"x": x.astype(str), "y": y})
                majority = frame.groupby("x")["y"].agg(lambda s: s.value_counts().iloc[0]).sum()
                purity = majority / len(frame)
                baseline = y.value_counts(normalize=True).iloc[0]
                if purity >= AUC_LIMIT and baseline < AUC_LIMIT:
                    out.append(_finding("UD004", os.path.basename(train_path),
                                        f"categories of '{col}' determine '{target}' for {purity:.1%} of rows"))
    return out
