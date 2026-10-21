<p align="center">
  <img src="assets/logo.svg" width="88" alt="unleak logo: a drop of data split into train and test">
</p>

<h1 align="center">unleak</h1>

<p align="center">
  <b>Find data leakage in machine-learning code before your metrics lie to you.</b><br>
  A static analyzer for Python scripts and Jupyter notebooks, plus an agent skill and a write hook<br>
  that stop Claude Code, Codex and Cursor from shipping leaky models.
</p>

<p align="center">
  <a href="https://pypi.org/project/unleak/"><img alt="PyPI" src="https://img.shields.io/pypi/v/unleak?color=0ea5e9"></a>
  <a href="https://pypi.org/project/unleak/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/unleak"></a>
  <a href="https://github.com/johnnyjosta-go/unleak/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/johnnyjosta-go/unleak/actions/workflows/ci.yml/badge.svg"></a>
  <a href="#use-it-with-your-coding-agent"><img alt="Claude Code plugin" src="https://img.shields.io/badge/Claude_Code-plugin-d97757"></a>
  <a href="#use-it-with-your-coding-agent"><img alt="Agent Skills" src="https://img.shields.io/badge/Agent_Skills-compatible-6366f1"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-22c55e"></a>
</p>

<p align="center"><b>English</b> · <a href="README.zh-CN.md">简体中文</a></p>

<p align="center"><img src="assets/demo.svg" alt="unleak check finds four leaks in a churn model: a feature built from the target, a median computed on all rows, SMOTE before the split, and early stopping on the test set" width="100%"></p>

Your imbalanced classifier scores **0.98 F1**. On data it has never seen: **0.21**. Nothing crashed,
no warning was printed: SMOTE simply ran one line before `train_test_split`.

That is **data leakage**: information from the evaluation rows reaches training, so the score measures
memory instead of skill. It is the most common way ML results fail to replicate. Kapoor & Narayanan
found it in [294 papers across 17 scientific fields](https://doi.org/10.1016/j.patter.2023.100804). It
is also all over tutorials, Kaggle notebooks and therefore the code AI assistants learned from.

**unleak** reads your code (it never runs it or touches your data) and points at the exact line where
test information leaks, with the reason and the fix. It runs in milliseconds per file, needs nothing but
Python 3.10+, and plugs into your coding agent so leaks are caught the moment they are written.

## What leaks cost (measured)

Each row is a runnable example in [`examples/`](examples): the same data, once with the leak and once
without. Mean of five seeds, reproduce with `python examples/run_all.py`.

| Leak | Rule | Metric | Leaky pipeline reports | Honest estimate |
|---|---|---|---:|---:|
| SMOTE before the split (3% positives) | UL103 | F1 | **0.978** | 0.207 |
| Classes balanced by hand before the split (fraud tutorial pattern) | UL104 | precision | **0.909** | 0.056 |
| Target mean encoding before the split, *no real signal* | UL302 | ROC AUC | **0.832** | 0.496 |
| Feature selection before CV, *pure noise* | UL102 | accuracy | **0.818** | 0.514 |
| Look-ahead feature (`shift(-1)`), *random-walk prices* | UL401 | accuracy | **0.727** | 0.490 |
| Shuffled split on a time series, *random walk* | UL402 | accuracy | **0.647** | 0.514 |
| Label defined from a column that stays a feature | UL303 | accuracy | **1.000** | 0.863 |
| Best of 50 models picked by test score, *no real signal* | UL204 | accuracy | **0.556** | 0.498 |
| Early stopping on the test set | UL203 | ROC AUC | **0.697** | 0.679 |
| `StandardScaler` before the split | UL101 | accuracy | **0.973** | 0.972 |

Not every leak is a disaster: the last row is the textbook one and here it barely matters, so unleak
rates it a *warning*. The others turn noise into a publishable result. Full table:
[`examples/RESULTS.md`](examples/RESULTS.md).

## Quick start

```bash
pip install unleak            # or run without installing: uvx unleak check .
unleak check train.py notebooks/
```

No pip at all? The analyzer is pure standard library:

```bash
git clone https://github.com/johnnyjosta-go/unleak
python3 unleak/skills/leakage-audit/scripts/run_unleak.py check path/to/your/project
```

Useful flags: `--format json|sarif|github|markdown`, `--min-severity info`, `--fail-on error`,
`--select UL1,UL301`, `--ignore UL106`. `unleak explain UL103` prints why a rule matters with a leaky
and a fixed example. Silence a deliberate case with `# unleak: ignore[UL103]` on the line.

## Use it with your coding agent

### Claude Code: plugin with skill + automatic check on every write

```
/plugin marketplace add johnnyjosta-go/unleak
/plugin install unleak@unleak
```

You get two things:

- **A write hook.** Every `.py` / `.ipynb` file Claude writes or edits is analyzed right away, and
  findings are handed back to Claude before it moves on:

  ```text
  unleak found possible data leakage in train.py:
  - line 9 [UL101 warning] StandardScaler.fit_transform() is fitted on rows that are later used for
    evaluation (rows are split into train/test here at line 10)
    fix: Split first, then fit preprocessing on the training rows only. ...
  ```

  Claude then explains the leak and moves the scaler into a `Pipeline` instead of reporting an
  inflated score. Disable it any time with `UNLEAK_HOOK=off`.

- **The `leakage-audit` skill.** Ask *"audit this notebook for data leakage"* (or run
  `/unleak:leakage-audit`) and Claude follows a six-step audit: map the pipeline across files, run the
  analyzer, check what static analysis cannot see (post-outcome features, duplicates, entity overlap),
  run data checks, verify each finding by re-running the evaluation, and report every finding with
  status, evidence, fix and the metric before and after.

### Codex, Cursor, Gemini CLI, OpenCode and other agents

The skill follows the open [Agent Skills](https://agentskills.io) format:

```bash
npx skills add johnnyjosta-go/unleak
```

or copy [`skills/leakage-audit`](skills/leakage-audit) into the skills folder your agent reads (for
Claude Code without the plugin: `~/.claude/skills/`; for Codex see its
[skills docs](https://developers.openai.com/codex/skills)). The skill carries its own copy of the
analyzer, so there is nothing else to install.

### Any agent: one line in `AGENTS.md` / `CLAUDE.md`

```markdown
Before reporting any model metric, run `uvx unleak check <files you changed>` and fix every finding.
```

## What it catches

| | Rule | |
|---|---|---|
| Preprocessing | UL101 · UL106 · UL107 | scalers, imputers, PCA, vectorizers, clustering or `df.mean()` fitted on rows later used for evaluation; fitting on train + test concatenated |
| Selection & resampling | UL102 · UL103 · UL104 · UL105 | feature selection or target encoding with test labels; SMOTE / undersampling before the split, including hand-rolled `resample` + `concat` |
| Test-set reuse | UL201 · UL202 · UL203 · UL204 | fitting on test data, training on data that contains the test rows, early stopping or model selection on the test set |
| Target leakage | UL301 · UL302 · UL303 | target left in `X`, features computed from the target, labels derived from a column that stays a feature |
| Time & groups | UL401 · UL402 · UL403 | `shift(-k)` / centred windows as features, shuffled splits on lagged data, entity ids without a grouped split |
| Real data | UD001–UD004 | `unleak data`: duplicate rows across splits, shared entities, temporal overlap, near-perfect single features |

Every rule with its reasoning and examples: [docs/rules.md](docs/rules.md).

It understands scikit-learn, imbalanced-learn, pandas, NumPy, XGBoost, LightGBM, CatBoost, Keras,
statsmodels and PyTorch's `random_split`; Jupyter notebooks (magics included); helper functions; and
data that one notebook saves with `to_csv`/`np.save` and another loads and splits.

## Check the data, not just the code

Some leaks only show in the data. Point `unleak data` at your actual split (CSV, Parquet, JSON, Excel;
needs `pip install "unleak[data]"`):

```bash
unleak data train.csv test.csv --target churned --group customer_id --time event_date
```

On the churn table that [`examples/make_leaky_split.py`](examples/make_leaky_split.py) splits row-wise
(a common mistake when customers have several rows):

```text
error    UD001  37 of 1184 test rows (3.1%) have the exact same feature values as a training row (e.g. test rows 1147, 1148, 1149, 1150, 1151)
error    UD002  760 'customer_id' values (e.g. 0, 100, 1003) appear in both files; 1061 test rows (89.6%) belong to entities seen in training
warning  UD003  3425 training rows (99.6%) are dated after the first test row (2024-01-01 00:00:00); the model can learn from the test period
warning  UD004  column 'days_since_cancellation' alone separates 'churned' with AUC 1.000
```

## In CI

GitHub Action, with annotations on the PR diff and a summary table:

```yaml
- uses: actions/checkout@v7
- uses: johnnyjosta-go/unleak@v0
  with:
    paths: notebooks src
    fail-on: error        # error | warning | info | never
```

[pre-commit](https://pre-commit.com):

```yaml
- repo: https://github.com/johnnyjosta-go/unleak
  rev: v0.1.0
  hooks:
    - id: unleak
```

GitHub code scanning: `unleak check . --format sarif -o unleak.sarif`, then upload it with
`github/codeql-action/upload-sarif`.

## How it works

unleak parses each file (notebooks cell by cell) and runs an abstract interpretation over the code. For
every variable it tracks:

- **which rows** it holds: which loaded dataset, and which side of which split (train, test, validation),
  following `train_test_split`, CV splitters, index arrays, masks, slices and naming (`X_test`);
- **which fitted steps** shaped it: every `fit`, `fit_transform`, `fit_resample`, `adapt` and pandas or
  NumPy statistic, and the rows that step saw;
- **which columns** a DataFrame still has, and how each derived column was computed (shifts, rolling
  windows, group-bys of the target).

A leak is then a precise question: did a step fitted on rows *without* a training tag shape data that
ends up on the evaluation side? Was the target, or a column it was computed from, still in `X` when
`fit(X, y)` ran? Everything is local and deterministic: no code execution, no data access, no network,
no LLM.

## Tested on real code

A linter that cries wolf gets uninstalled, so unleak is tuned for precision. We ran it on 499
real-world files and read every finding:

| Corpus | Files | Findings | Real leaks |
|---|---:|---:|---:|
| scikit-learn 1.9 example gallery | 281 | 7 warnings | 7 |
| imbalanced-learn examples | 22 | 0 | – |
| Notebooks of two well-known ML books | 16 | 2 warnings | 2 |
| Two popular ML course repositories (incl. translated copies) | 176 | 1 error | 1 |
| Kaggle-style notebooks collected from GitHub | 4 | 2 errors, 7 warnings | 9 |

Some of these are mild, for example a scaler fitted on all of `X_train` before cross-validation. One
of scikit-learn's own examples scales all data before a grid search, and its comment says it is
*"cheating a bit"*; unleak flags exactly that line. One course leak spans two notebooks: the first
runs SMOTE and saves a CSV, the second loads it and splits. You can only see it by following data
across files.

## Limitations

- **Static analysis sees code, not meaning.** A column recorded after the outcome (`refund_issued`
  when predicting churn) looks like any other feature. The audit skill and `unleak data` (UD004) exist
  for this; a clean report is not proof of no leakage.
- **Heuristics.** unleak follows data through assignments, method chains, helper functions and saved
  files, and reads names like `X_test`. Very dynamic code (data built in loops over `getattr`, models
  passed through config objects) can hide a flow. Please [report misses](../../issues/new?template=missed-leak.yml)
  and [false positives](../../issues/new?template=false-positive.yml).
- **Python only.** R, Spark ML and SQL feature pipelines are out of scope for now.

## FAQ

**Is scaling before the split really a leak?** Yes, but usually a small one; see the last row of the
table. Imputers, PCA, TF-IDF and clustering on small datasets can be much worse, so they share the
rule and the warning level.

**Kaggle: is fitting on train + test allowed?** Using unlabeled test features is a known competition
trick. UL107 warns because any validation score computed from rows that went into that fit is
optimistic.

**Does it send my code anywhere?** No. It is a local static analyzer with no dependencies.

## Contributing

Ideas, false-positive reports and new rules are very welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
Each rule ships with positive and negative cases in [`tests/cases`](tests/cases), where a comment like
`# expect: UL103` marks the line that must be flagged.

## Acknowledgements

The rule set builds on the leakage taxonomy of Kapoor & Narayanan,
[*Leakage and the reproducibility crisis in machine-learning-based science*](https://doi.org/10.1016/j.patter.2023.100804)
(Patterns, 2023), and on Yang et al.,
[*Data Leakage in Notebooks: Static Detection and Better Processes*](https://arxiv.org/abs/2209.03345) (ASE 2022).

## License

MIT
