# Changelog

## 0.1.0

First release.

- Static analyzer for `.py` and `.ipynb` with 17 rules: preprocessing, feature selection,
  resampling and target encoding before the split (UL101–UL107); test-set reuse (UL201–UL204);
  target leakage (UL301–UL303); look-ahead features, shuffled time-series splits and ungrouped
  entities (UL401–UL403).
- Follows data through helper functions and across files saved with `to_csv` / `np.save` /
  `joblib.dump` and loaded elsewhere.
- `unleak data`: duplicate rows, entity overlap, temporal overlap and near-perfect single features
  on real train/test files (UD001–UD004).
- Output as text, JSON, SARIF, GitHub annotations or Markdown; inline `# unleak: ignore[...]`.
- `leakage-audit` agent skill (Agent Skills format) and a Claude Code plugin with a PostToolUse
  hook that checks every file Claude writes.
- GitHub Action and pre-commit hook.
- Ten examples measuring how much each leak inflates metrics.
