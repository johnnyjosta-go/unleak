# Contributing to unleak

Thanks for helping make ML results more trustworthy. The most valuable contributions are:

1. **False-positive reports**: a minimal snippet that unleak flags but that does not leak.
2. **Missed leaks**: a minimal snippet that leaks while unleak stays silent.
3. **New rules** for leakage patterns we do not cover yet.

## Development setup

```bash
git clone https://github.com/OWNER/unleak && cd unleak
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # core tests, Python 3.10+
pip install scikit-learn imbalanced-learn lightgbm xgboost
pytest tests/test_examples.py            # every example is flagged on its leaky line only
python examples/run_all.py               # regenerate examples/RESULTS.md
```

## Where things live

```
skills/leakage-audit/              the agent skill (self-contained: agents run it with python3, no install)
├── SKILL.md                       audit workflow and writing rules for coding agents
├── references/                    taxonomy, fix patterns, report template
└── scripts/unleak/                the Python package (pip installs it from here)
    ├── engine.py                  abstract interpretation: rows, splits, fitted steps, columns
    ├── knowledge.py               what unleak knows about sklearn, imblearn, pandas, xgboost, keras...
    ├── rules.py                   rule ids, severities and explanations
    ├── checker.py                 files -> findings, cross-file data flow, suppressions
    ├── report.py / cli.py         output formats and the command line
    ├── data.py                    `unleak data` checks on real train/test files
    └── hook.py                    Claude Code PostToolUse hook
.claude-plugin/, hooks/            Claude Code plugin and marketplace
tests/cases/                       annotated case files: `# expect: UL103` marks a required finding
examples/                          leaky vs honest pipelines with measured inflation
```

## How the analysis works, in one paragraph

`engine.Scope` walks a module or function in order and keeps a `Val` per variable: the
datasets its rows come from (`roots`), which side of which split they are on (`tags`), the fitted
steps that shaped them (`fits`) and, for DataFrames, which columns they hold (`frame`). Splits create
tags; `fit*` calls and pandas/NumPy statistics create `FitEvent`s. At the end, `Analyzer._finalize`
asks whether any step fitted on rows without a training tag shaped data that later shows up on the
evaluation side. Target leakage is checked where a model receives `(X, y)`.

## Adding or changing a rule

1. Add the rule text to `rules.py` (id, name, severity, why, fix, leaky and fixed example).
2. Implement detection in `engine.py`, usually by creating a `FitEvent` or reporting at a pair site.
3. Add **positive and negative** case files to `tests/cases/`. Mark expected lines with
   `# expect: ULxxx`. A file without markers must produce no findings.
4. Run `python tools/gen_docs.py` to refresh `docs/rules.md`, and `pytest`.
5. Precision matters more than recall. Before opening the PR, run the new rule over a real corpus
   (for example the scikit-learn example gallery: `unleak check path/to/scikit-learn/examples`) and
   read every new finding. Mention the result in the PR.

## Style

`ruff check skills tests tools` must pass. Keep the core free of third-party dependencies: the
analyzer has to run from the skill folder with a bare `python3`.
