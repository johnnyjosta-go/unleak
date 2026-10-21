# Leakage audit report template

Use this structure when reporting an audit. Keep evidence concrete: file, line or
cell, and the observed metric change.

```markdown
## Leakage audit: <project / notebook>

**Scope:** <files and data checked>  ·  **Unit of independence:** <row / patient / user / time>
**Analyzer:** unleak <version>, `unleak check <paths>`: <N> findings (<e> errors, <w> warnings)

### Findings

| # | Rule | Where | Status | Evidence | Fix | Metric before → after |
|---|------|-------|--------|----------|-----|------------------------|
| 1 | UL103 | prepare.ipynb cell 16 | confirmed | SMOTE runs on all rows; data saved to balanced.csv and split in train.ipynb cell 5 | resample inside imblearn Pipeline on training folds | F1 0.98 → 0.27 |
| 2 | UL101 | train.py:41 | confirmed (small) | StandardScaler fitted before split | moved into Pipeline | acc 0.973 → 0.972 |
| 3 | — | features.sql | plausible | `days_since_last_claim` computed over full history | compute as-of the prediction date | not re-run |
| 4 | UL403 | train.py:52 | rejected | patient_id is unique per row (`df.patient_id.is_unique`) | `# unleak: ignore[UL403] one row per patient` | — |

### Checked and clean
- Target and post-outcome columns: reviewed all <k> features against the data dictionary
- Duplicates across splits: `unleak data` UD001 clean
- Test set used once: yes / no (explain)

### Corrected result
<The honest metric, how it was obtained, and remaining caveats.>
```

Status meanings:
- **confirmed**: the metric changes when fixed, or the mechanism is certain
- **plausible**: clear mechanism, not re-run (say why)
- **rejected**: safe in this context, with the reason recorded in the ignore comment
