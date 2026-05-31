# Implementation Status

Implemented stages:

- WRDS schema audit with frozen table contracts.
- Sharded WRDS extraction for the 2006-2025 development window.
- Point-in-time monthly modeling panel construction with filing-date activation
  support.
- Cached macro tensor builder with vintage-timing flags.
- Targeted filing-date supplement run `20260531T003829Z_filing` and activated
  panel run `20260531T003936Z_panel_filing`.
- First-pass exposure-sorted portfolios, rank ICs, and cross-sectional slopes.
- Expanding-window LightGBM benchmark code, Slurm runner, and full-panel
  filing-date ablation run `20260531T005001Z_lgbm_filing`.
- Deep Sets segment-set model code, Slurm runner, and full-panel run
  `20260531T010832Z_set`.
- Full CUDA Set Transformer segment-set diagnostic.
- Factor-alpha and turnover robustness runner from cached model predictions.
- Claim-ledger and table-inventory generator for wording discipline.
- Publication-style diagnostic table renderer for model comparison,
  factor-alpha, and turnover-cost outputs.
- Visual pack and model-card generator, with private visual pack run
  `20260531T_visual_pack`.
- Public-safe release audit script and release notes.
- Requirement-by-requirement completion audit.
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
- Full `set_transformer` run `20260531T_set_transformer_full` completed on
  CUDA with 3 uncapped epochs, 772,000 prediction rows, mean rank IC
  `0.007315`, and mean Q5-Q1 `0.000426`. Treat as a full-scale architecture
  diagnostic, not as evidence that attention dominates the tabular benchmark.

Latest private factor robustness diagnostic:

- Run: `20260531T_factor_robustness_with_transformer`.
- Panel run: `20260531T003936Z_panel_filing`.
- Model runs: `20260531T005001Z_lgbm_filing`, `20260531T010832Z_set`, and
  `20260531T_set_transformer_full`.
- Prediction rows evaluated: `6176000`.
- Variants evaluated: `8`.
- Spread-month rows: `1528`.
- Factor months available: `239`.
- Cost assumption: `10` bps per one-way turnover.
- Best net Q5-Q1 variant: `lgbm:non_segment_controls`, mean net Q5-Q1
  `0.006624`, t-stat `2.155836`.
- Best factor-alpha variant: `lgbm:no_return_or_market`, gross monthly alpha
  `0.006141`, t-stat `2.582860`.
- Segment-only LightGBM robustness: mean net Q5-Q1 `0.003756`, t-stat
  `1.920601`, gross monthly alpha `0.001023`.
- Full Set Transformer robustness: mean net Q5-Q1 `0.000367`, t-stat
  `0.282725`, gross monthly alpha `-0.003558`.

Latest private claim ledger:

- Run: `20260531T_claim_ledger_with_transformer`.
- Claim rows: `7`.
- Validation failures: `0`.
- Blocked claims: `1` for live macro execution.
- Table-inventory rows: `40`.
- Allowed wording is diagnostic-only and keeps macro-vintage and 2026 holdout
  claims blocked.

Latest private publication-style diagnostic tables:

- Run: `20260531T_publication_tables_with_transformer_v2`.
- Inputs: panel `20260531T003936Z_panel_filing`, LightGBM
  `20260531T005001Z_lgbm_filing`, Deep Sets `20260531T010832Z_set`, full Set
  Transformer `20260531T_set_transformer_full`, factor robustness
  `20260531T_factor_robustness_with_transformer`, and claim ledger
  `20260531T_claim_ledger_with_transformer`.
- Model-comparison rows: `8`.
- Factor-alpha and cost rows: `8`.
- Review failures: `0`.
- Claim-validation failures: `0`.
- Reports generated as ignored private Markdown/LaTeX artifacts under `runs/`;
  table CSVs generated under ignored `artifacts/tables/`.

Latest private visual pack:

- Run: `20260531T_visual_pack_with_transformer_v2`.
- Figure count: `7`.
- Model comparison rows: `8`.
- Firm explorer rows: `30`.
- Sector-geography matrix shape: `10 x 10`.
- Dashboard: ignored private HTML artifact under `artifacts/figures_html/`.
- Model card: ignored private report under `runs/`.

Latest public release-prep check:

- Hard-coded private Amarel path and allocation id removed from public runner
  scripts and docs.
- Runners require `SMB_PROJECT_ROOT` and `SMB_SLURM_JOB_ID`.
- CI now runs the public safety scan, release audit, and unit tests.
- Local checks passed: release audit, public safety scan, and 58 unit tests.
- Allocation-backed checks passed on the active compute environment: release
  audit, public safety scan, and 58 unit tests.
- Macro-engine runner dry run `20260531T_release_macro_dry` completed with
  status `dry_run_ok`; no API credentials were present and no API calls were
  executed.

Latest macro-tensor code status:

- Public macro-series catalog is in `configs/macro_series.yml` with configured
  release-lag timing metadata and FRED/BLS/BEA/EIA source adapters.
- Public code can construct firm-month macro interaction features and
  segment-token macro tensors from cached private macro Parquet files.
- Macro timing uses `available_date`, `realtime_start`, or `release_date` when
  present; observation-date fallback is explicitly flagged as not fully
  vintage-safe.
- Global macro states are allowed as fallback states for all segment-token
  areas.
- Macro-engine multi-source catalog dry run `20260531T_macro_multisource_dry`
  passed locally with no credentials present and no API calls executed.
- Execute-mode missing-secret guard run
  `20260531T_macro_multisource_missing_guard` stopped before any API call with
  status `missing_credentials`.
- Full macro API execution is still gated on an untracked compute-host `.env`.

Latest GPU modeling code status:

- LightGBM runner now has `LGBM_DEVICE_TYPE=auto`, attempting GPU training and
  recording any CPU fallback in the manifest.
- Deep Sets runner now has `SET_DEVICE_TYPE=auto`, recording the selected
  PyTorch device and CUDA device name in the manifest.
- The PyTorch set-model DataLoader pins host memory when CUDA is selected.
- Allocation probe `20260531T_gpu_probe`: PyTorch selected `cuda` on the A100
  and completed a tiny CUDA matmul. The current `ml_core` LightGBM build does
  not enable the GPU tree learner, so LightGBM auto mode records a CPU
  fallback.
- Allocation gate `20260531T_gpu_gate2`: CUDA assertion passed with
  `pin_memory=true` for the set-model DataLoader.

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
- Interpret factor robustness carefully: no-return-or-market and
  no-market-factor LightGBM variants have the strongest alpha diagnostics,
  while segment-only features remain weaker after factor adjustment. The
  publication-style tables remain diagnostic and are not final paper claims.
- Interpret the Deep Sets benchmark carefully: the simple set-only encoder has
  positive rank IC but weak long-short spread, and the first controls variant
  does not improve rank IC.
- Interpret the full Set Transformer carefully: it runs on CUDA but does not
  improve the economic spread diagnostics in the current development sample.
- Push only after explicit user approval and a final clean audit.
