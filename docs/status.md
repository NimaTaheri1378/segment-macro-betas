# Implementation Status

Implemented stages:

- WRDS schema audit with frozen table contracts.
- Sharded WRDS extraction for the 2006-2025 development window.
- Point-in-time monthly modeling panel construction.
- First-pass exposure-sorted portfolios, rank ICs, and cross-sectional slopes.
- Expanding-window LightGBM benchmark code, Slurm runner, and full-panel run
  `20260531T000937Z`.
- Smoke and baseline figures generated from private artifacts.

Latest private LightGBM diagnostic:

- Panel run: `20260530T234643Z`.
- Prediction rows: `778755`.
- Rank IC months: `191`.
- Mean monthly rank IC: `0.020776`.
- Mean predicted Q5-Q1 next-month excess return: `0.007652`.

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
- LightGBM ablations that remove same-month market factors and isolate segment
  exposure contribution.
- Segment-set model extension.
- Publication-safe figure and model-card selection.
