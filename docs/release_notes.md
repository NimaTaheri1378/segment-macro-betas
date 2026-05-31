# Release Notes

This repository is prepared for a public code release, but private WRDS data,
run logs, tables, figures, dashboards, credentials, and local environment files
remain excluded.

Implemented public code:

- WRDS schema audit and frozen schema map.
- Restartable sharded WRDS extraction.
- Filing-date activation supplement and point-in-time monthly panel builder.
- Baseline portfolio and cross-sectional diagnostics.
- Cached macro tensor construction with vintage-timing flags.
- Expanding-window LightGBM benchmark with feature ablations.
- Deep Sets segment-set benchmark.
- Cached factor-alpha and turnover robustness diagnostics.
- GPU-aware model runners: PyTorch set models use CUDA when available, and
  LightGBM attempts GPU training with an explicit manifest fallback if the
  installed build lacks GPU support.
- Static visual pack, HTML dashboard, and model-card generator.
- Public safety and release audit checks.
- Requirement-by-requirement completion audit in `docs/completion_audit.md`.

Private empirical artifacts are intentionally not part of the public release.
The current private diagnostics are summarized in `docs/status.md` as research
status notes, not as final paper claims.

Known gated items:

- Macro API execution requires an untracked compute-host `.env`.
- Full vintage-safe macro interactions require private cached macro files with
  release or realtime availability dates.
- Set Transformer, final reviewed factor-alpha tables, transaction-cost
  specifications, and 2026 holdout evaluation remain future stages.

Before pushing, run:

```bash
python scripts/public_safety_scan.py
python scripts/release_audit.py
python -m unittest discover -s tests
```
