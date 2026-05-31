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
- Deep Sets segment-set model code, Slurm runner, and full-panel run
  `20260531T010832Z_set`.
- Visual pack and model-card generator, with private visual pack run
  `20260531T_visual_pack`.
- Public-safe release audit script and release notes.
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

Latest private Deep Sets segment-set diagnostic:

- Panel run: `20260531T003936Z_panel_filing`.
- Raw segment run: `20260530T233446Z`.
- Encoded panel rows: `926895`.
- Matched set rows: `926895`.
- Token snapshots: `101958`.
- Vocabulary size: `312`.
- Prediction rows per variant: `772000`.
- `set_only`: mean rank IC `0.011294`, mean Q5-Q1 `0.001139`.
- `set_plus_controls`: mean rank IC `-0.006154`, mean Q5-Q1 `0.002959`.

Latest private visual pack:

- Run: `20260531T_visual_pack`.
- Figure count: `7`.
- Model comparison rows: `7`.
- Firm explorer rows: `30`.
- Sector-geography matrix shape: `10 x 10`.
- Dashboard: ignored private HTML artifact under `artifacts/figures_html/`.
- Model card: ignored private report under `runs/`.

Latest public release-prep check:

- Hard-coded private Amarel path and allocation id removed from public runner
  scripts and docs.
- Runners require `SMB_PROJECT_ROOT` and `SMB_SLURM_JOB_ID`.
- Local checks passed: release audit, public safety scan, and 32 unit tests.
- Allocation-backed checks passed on the active compute environment: release
  audit, public safety scan, and 32 unit tests.
- Macro-engine runner dry run `20260531T_release_macro_dry` completed with
  status `dry_run_ok`; no API credentials were present and no API calls were
  executed.

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
- Interpret the Deep Sets benchmark carefully: the simple set-only encoder has
  positive rank IC but weak long-short spread, and the first controls variant
  does not improve rank IC.
- Push only after explicit user approval and a final clean audit.
