"""Run every example over several seeds and write RESULTS.md.

    pip install scikit-learn imbalanced-learn lightgbm pandas
    python examples/run_all.py
"""

import importlib
import pathlib
import platform
import statistics
import sys
import warnings

import sklearn

HERE = pathlib.Path(__file__).parent
EXAMPLES = [
    "feature_selection_before_cv",
    "smote_before_split",
    "manual_undersampling",
    "target_mean_encoding",
    "target_proxy",
    "lookahead_feature",
    "shuffled_time_series_split",
    "early_stopping_on_test",
    "tuning_on_test_set",
    "scaler_before_split",
]
# These two compare the score the leaky pipeline reports with the same model on fresh data.
SAME_MODEL = {"early_stopping_on_test", "tuning_on_test_set"}


def main(seeds: int = 5) -> None:
    sys.path.insert(0, str(HERE))
    warnings.filterwarnings("ignore")
    rows = []
    for name in EXAMPLES:
        mod = importlib.import_module(name)
        results = [mod.run(seed) for seed in range(seeds)]
        leaky = statistics.mean(r[0] for r in results)
        honest = statistics.mean(r[1] for r in results)
        rows.append((name, mod, leaky, honest))
        print(f"{name:32s} leaky={leaky:.3f} honest={honest:.3f}", flush=True)

    lines = [
        "# What each leak does to the numbers",
        "",
        f"Mean over {seeds} random seeds. Every leaky pipeline below is flagged by `unleak check examples/`;",
        "every honest one passes. Reproduce with `python examples/run_all.py`.",
        "",
        "| Example | Rule | Metric | Leaky pipeline reports | Honest estimate |",
        "|---|---|---|---:|---:|",
    ]
    for name, mod, leaky, honest in rows:
        lines.append(f"| [{mod.TITLE}]({name}.py) | {mod.RULE} | {mod.METRIC} | **{leaky:.3f}** | {honest:.3f} |")
    lines += [
        "",
        "For early stopping and model selection on the test set, the honest estimate is the *same chosen "
        "model* scored on fresh rows that played no part in training or selection.",
        "",
        f"Environment: Python {platform.python_version()}, scikit-learn {sklearn.__version__}.",
        "",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines))
    print("wrote", HERE / "RESULTS.md")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
