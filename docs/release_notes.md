# Release Notes

This repository is prepared for a public code release, but private WRDS data,
run logs, tables, figures, dashboards, credentials, and local environment files
remain excluded.

Implemented public code:

- WRDS schema audit and frozen schema map.
- Restartable sharded WRDS extraction.
- Filing-date activation supplement and point-in-time monthly panel builder.
- Baseline portfolio and cross-sectional diagnostics.
- Expanding-window LightGBM benchmark with feature ablations.
- Deep Sets segment-set benchmark.
- Static visual pack, HTML dashboard, and model-card generator.
- Public safety and release audit checks.

Private empirical artifacts are intentionally not part of the public release.
The current private diagnostics are summarized in `docs/status.md` as research
status notes, not as final paper claims.

Known gated items:

- Macro API execution requires an untracked compute-host `.env`.
- Vintage-safe macro interactions and the firm-geography-macro tensor remain a
  gated extension.
- Set Transformer, factor alphas, transaction-cost tests, and 2026 holdout
  evaluation remain future stages.

Before pushing, run:

```bash
python scripts/public_safety_scan.py
python scripts/release_audit.py
python -m unittest discover -s tests
```
