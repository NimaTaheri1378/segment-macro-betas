# Implementation Status

Implemented stages:

- WRDS schema audit with frozen table contracts.
- Sharded WRDS extraction for the 2006-2025 development window.
- Point-in-time monthly modeling panel construction with filing-date activation
  support.
- Targeted filing-date supplement run `20260531T003829Z_filing` and activated
  panel run `20260531T003936Z_panel_filing`.
- First-pass exposure-sorted portfolios, rank ICs, and cross-sectional slopes.
- Expanding-window LightGBM benchmark code, Slurm runner, and full-panel
  filing-date ablation run `20260531T005001Z_lgbm_filing`.
- Smoke and baseline figures generated from private artifacts.

Latest private LightGBM ablation diagnostic:

- Panel run: `20260531T003936Z_panel_filing`.
- Panel rows: `936897`.
- Filing-date match rate: `0.990869`.
- Activation-rule violations: `0`.
- Prediction rows: `772000`.
- Rank IC months: `191`.
- Variants completed: `5`.
- Best rank-IC variant: `segment_only`, mean rank IC `0.036475`.
- Best Q5-Q1 spread variant: `non_segment_controls`, mean Q5-Q1 `0.007281`.
- Full-feature variant: mean rank IC `0.022796`, mean Q5-Q1 `0.007274`.

Private artifacts remain ignored:

- `runs/`
- `data/`
- `artifacts/tables/`
- `artifacts/figures_static/`
- `artifacts/figures_html/`

Next stages:

- Macro vintage engine execution once local secret handling is enabled on the
  compute host.
- Interpret the LightGBM ablation carefully: segment-only features rank returns
  well, while full/non-segment-control variants currently produce the stronger
  long-short spread.
- Segment-set model extension.
- Publication-safe figure and model-card selection.
