# Release Notes

This repository is prepared for a public code release, but private WRDS data,
run logs, tables, figures, dashboards, credentials, and local environment files
remain excluded.

Implemented public code:

- WRDS schema audit and frozen schema map.
- Restartable sharded WRDS extraction.
- Filing-date activation supplement and point-in-time monthly panel builder.
- Baseline portfolio and cross-sectional diagnostics.
- Public FRED/BLS/BEA/EIA macro-series catalog plus cached macro tensor
  construction with timing flags.
- Expanding-window LightGBM benchmark with feature ablations and optional
  macro-aware panel datasets.
- Deep Sets segment-set benchmark plus full CUDA Set Transformer diagnostic.
- Cached factor-alpha and turnover robustness diagnostics.
- Publication-style private diagnostic table renderer for model comparison,
  factor-alpha, and turnover-cost outputs.
- GPU-aware model runners: PyTorch set models use CUDA when available, and
  LightGBM attempts GPU training with an explicit manifest fallback if the
  installed build lacks GPU support.
- Static visual pack, HTML dashboard, and model-card generator.
- Public safety and release audit checks.
- GitHub CI runs public safety scan, release audit, and unit tests.
- GitHub release checklist in `docs/github_release_checklist.md`.
- Claim guardrails and private claim-ledger generator.
- Requirement-by-requirement completion audit in `docs/completion_audit.md`.

Private empirical artifacts are intentionally not part of the public release.
The current private diagnostics are summarized in `docs/status.md` as research
status notes, not as final paper claims.

Known gated items:

- Official non-FRED macro execution has run privately from an untracked
  compute-host `.env`; the full FRED-inclusive catalog is currently blocked by
  HTTP `429` rate limiting.
- Full revision-safe macro interactions require private cached macro files with
  true vintage or realtime availability dates.
- Final transaction-cost/capacity claims and 2026 holdout evaluation remain
  future stages.

Before pushing, run:

```bash
python scripts/public_safety_scan.py
python scripts/release_audit.py
python -m unittest discover -s tests
```
