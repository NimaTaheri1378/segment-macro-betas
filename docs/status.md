# Implementation Status

Implemented stages:

- WRDS schema audit with frozen table contracts.
- Sharded WRDS extraction for the 2006-2025 development window.
- Point-in-time monthly modeling panel construction.
- First-pass exposure-sorted portfolios, rank ICs, and cross-sectional slopes.
- Expanding-window LightGBM benchmark code, Slurm runner, and full-panel
  ablation run `20260531T002047Z`.
- Smoke and baseline figures generated from private artifacts.

Latest private LightGBM ablation diagnostic:

- Panel run: `20260530T234643Z`.
- Prediction rows: `778755`.
- Rank IC months: `191`.
- Variants completed: `5`.
- Best rank-IC variant: `segment_only`, mean rank IC `0.036423`.
- Best Q5-Q1 spread variant: `non_segment_controls`, mean Q5-Q1 `0.008149`.
- Full-feature variant: mean rank IC `0.020776`, mean Q5-Q1 `0.007652`.

Private artifacts remain ignored:

- `runs/`
- `data/`
- `artifacts/tables/`
- `artifacts/figures_static/`
- `artifacts/figures_html/`

Next stages:

- Macro vintage engine execution once local secret handling is enabled on the
  compute host.
- SEC filing-date activation upgrade.
- Interpret the LightGBM ablation carefully: segment-only features rank returns
  well, but non-segment controls currently produce the stronger long-short
  spread.
- Segment-set model extension.
- Publication-safe figure and model-card selection.
