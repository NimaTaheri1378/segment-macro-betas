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
- Official non-FRED macro API execution, macro tensor construction, and
  macro-aware LightGBM diagnostics.
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

Latest private official macro diagnostic:

- Full FRED/BLS/BEA/EIA catalog run:
  `20260531T_macro_full_catalog_delayed`.
- Sources: FRED, BLS, BEA, and EIA.
- Series: `8`.
- Macro rows: `8658`.
- Window: 2006-2025 development sample.
- Timing flags: `lookahead_safe=true`, `revision_safe=false`.
- Request discipline: one delayed full-catalog retry with
  `MACRO_REQUEST_DELAY_SECONDS=5`; no API errors and no credential values in
  manifests.
- Non-FRED official macro run: `20260531T_macro_nonfred_full`.
- Sources: BLS, BEA, and EIA.
- Series: unemployment rate, GDP growth, and WTI crude oil price.
- Macro rows: `2480`.
- Window: 2006-2025 development sample.
- Timing flags: `lookahead_safe=true`, `revision_safe=false`.

Latest private FRED initial-release diagnostic:

- Macro run: `20260531T_fred_initial_release_guarded`.
- Sources: FRED only.
- Series: initial-release CPI and unemployment.
- Macro rows: `480`.
- Window: 2006-2025 development sample.
- Timing flags: `lookahead_safe=true`, `revision_safe=true`.
- Request discipline: one guarded run with `MACRO_REQUEST_DELAY_SECONDS=2`;
  no API errors and no credential values in manifests.

Latest private macro tensor diagnostic:

- Run: `20260531T_macro_tensor_nonfred_v2`.
- Macro run: `20260531T_macro_nonfred_full`.
- Panel run: `20260531T003936Z_panel_filing`.
- Panel rows: `936897`.
- Joined token rows: `2811167`.
- Joined token match rate: `1.0`.
- Macro feature count: `9`.
- Macro coverage rate: `1.0`.
- Macro availability source: `available_date`.
- Timing flags: `lookahead_safe=true`, `revision_safe=false`.

Latest private full-catalog macro tensor:

- Run: `20260531T_macro_tensor_full_catalog`.
- Macro run: `20260531T_macro_full_catalog_delayed`.
- Panel run: `20260531T003936Z_panel_filing`.
- Panel rows: `936897`.
- Joined token rows: `2811167`.
- Joined token match rate: `1.0`.
- Macro feature count: `24`.
- Macro coverage rate: `1.0`.
- Macro availability source: `available_date`.
- Timing flags: configured no-lookahead availability; not true full-catalog
  revision safety.

Latest private revision-safe FRED macro tensor:

- Run: `20260531T_macro_tensor_fred_initial_release`.
- Macro run: `20260531T_fred_initial_release_guarded`.
- Panel run: `20260531T003936Z_panel_filing`.
- Panel rows: `936897`.
- Joined token rows: `2811167`.
- Joined token match rate: `1.0`.
- Macro feature count: `6`.
- Macro coverage rate: `1.0`.
- Macro availability source: `available_date`.
- Timing flag: `vintage_safe=true`.

Latest private macro-aware LightGBM diagnostic:

- Run: `20260531T_lgbm_macro_nonfred_v3`.
- Panel dataset: `macro_tensor_panel` from
  `20260531T_macro_tensor_nonfred_v2`.
- Feature rows: `926895`.
- Prediction rows per variant: `772000`.
- Variants: 4 cross-sectional variants `ok`, plus `macro_only` marked
  `diagnostic_only` because global macro features have no within-month
  cross-sectional ranking variation.
- Best rank-IC variant: `segment_only`, mean rank IC `0.036475`.
- Best Q5-Q1 spread variant: `all_plus_macro`, mean Q5-Q1 `0.010795`,
  t-stat `3.718524`.
- `all_plus_macro`: mean rank IC `0.026850`, t-stat `3.807517`.
- `segment_plus_macro`: mean rank IC `0.017409`, mean Q5-Q1 `0.003772`.
- LightGBM requested GPU in auto mode, detected the installed build lacks the
  GPU tree learner, disabled further GPU attempts after the first failure per
  variant, and recorded CPU fallback in the manifest.

Latest private full-catalog LightGBM diagnostic:

- Run: `20260531T_lgbm_full_catalog`.
- Panel dataset: `macro_tensor_panel` from
  `20260531T_macro_tensor_full_catalog`.
- Feature rows: `926895`.
- Prediction rows per variant: `772000`.
- Variants: 4 cross-sectional variants `ok`, plus `macro_only` marked
  `diagnostic_only`.
- Best rank-IC variant: `segment_only`, mean rank IC `0.036475`.
- Best Q5-Q1 spread variant: `all_plus_macro`, mean Q5-Q1 `0.007953`,
  t-stat `2.626401`.
- `all_plus_macro`: mean rank IC `0.016630`, t-stat `2.114252`.

Latest private revision-safe FRED LightGBM diagnostic:

- Run: `20260531T_lgbm_fred_initial_release`.
- Panel dataset: `macro_tensor_panel` from
  `20260531T_macro_tensor_fred_initial_release`.
- Feature rows: `926895`.
- Prediction rows per variant: `772000`.
- Variants: 4 cross-sectional variants `ok`, plus `macro_only` marked
  `diagnostic_only`.
- Best rank-IC variant: `segment_only`, mean rank IC `0.036475`.
- Best Q5-Q1 spread variant: `all_plus_macro`, mean Q5-Q1 `0.008831`,
  t-stat `2.848604`.
- `all_plus_macro`: mean rank IC `0.024862`, t-stat `3.257238`.

Latest private macro-aware factor robustness diagnostic:

- Run: `20260531T_factor_robustness_macro_nonfred`.
- Panel run: `20260531T003936Z_panel_filing`.
- Model runs: `20260531T_lgbm_macro_nonfred_v3`,
  `20260531T010832Z_set`, and `20260531T_set_transformer_full`.
- Prediction rows evaluated: `6176000`.
- Spread-capable variants evaluated: `7`.
- Spread-month rows: `1337`.
- Factor months available: `239`.
- Cost assumption: `10` bps per one-way turnover.
- Best net Q5-Q1 variant: `lgbm:all_plus_macro`, mean net Q5-Q1
  `0.010245`, t-stat `3.528957`.
- Best factor-alpha variant: `lgbm:all_plus_macro`, gross monthly alpha
  `0.008763`, t-stat `2.962949`.
- Segment-only LightGBM robustness: mean net Q5-Q1 `0.003756`, t-stat
  `1.920601`, gross monthly alpha `0.001023`.
- Full Set Transformer robustness: mean net Q5-Q1 `0.000367`, t-stat
  `0.282725`, gross monthly alpha `-0.003558`.

Latest private full-catalog factor robustness diagnostic:

- Run: `20260531T_factor_robustness_full_catalog`.
- Model runs: `20260531T_lgbm_full_catalog`, `20260531T010832Z_set`, and
  `20260531T_set_transformer_full`.
- Spread-capable variants evaluated: `7`.
- Spread-month rows: `1337`.
- Factor months available: `239`.
- Best net Q5-Q1 variant: `lgbm:all_plus_macro`, mean net Q5-Q1
  `0.007445`, t-stat `2.459427`.
- Best gross alpha t-stat variant: `lgbm:non_segment_controls`, gross monthly
  alpha `0.005865`, t-stat `1.676543`.

Latest private revision-safe FRED factor robustness diagnostic:

- Run: `20260531T_factor_robustness_fred_initial_release`.
- Model runs: `20260531T_lgbm_fred_initial_release`,
  `20260531T010832Z_set`, and `20260531T_set_transformer_full`.
- Spread-capable variants evaluated: `7`.
- Spread-month rows: `1337`.
- Factor months available: `239`.
- Best net Q5-Q1 variant: `lgbm:all_plus_macro`, mean net Q5-Q1
  `0.008304`, t-stat `2.678754`.
- Best gross alpha t-stat variant: `lgbm:non_segment_controls`, gross monthly
  alpha `0.005865`, t-stat `1.676543`.

Latest private claim ledger:

- Run: `20260531T_claim_ledger_macro_nonfred`.
- Claim rows: `7`.
- Validation failures: `0`.
- Blocked claims: `1` for final macro-vintage or holdout wording.
- Table-inventory rows: `40`.
- Allowed wording is diagnostic-only and keeps macro-vintage and 2026 holdout
  claims blocked.

Latest private full-catalog claim ledger:

- Run: `20260531T_claim_ledger_full_catalog_v2`.
- Claim rows: `7`.
- Validation failures: `0`.
- Blocked claims: `0`.
- Allowed macro wording says the full official catalog has been pulled and
  joined with configured no-lookahead availability dates; it does not call the
  full catalog true historical-vintage evidence.

Latest private revision-safe FRED claim ledger:

- Run: `20260531T_claim_ledger_fred_initial_release_v2`.
- Claim rows: `7`.
- Validation failures: `0`.
- Blocked claims: `0`.
- Allowed macro wording is limited to the included FRED initial-release series;
  full-catalog no-lookahead diagnostics are tracked separately and 2026
  holdout performance remains unopened.

Latest private publication-style diagnostic tables:

- Run: `20260531T_publication_tables_macro_nonfred`.
- Inputs: panel `20260531T003936Z_panel_filing`, LightGBM
  `20260531T_lgbm_macro_nonfred_v3`, Deep Sets `20260531T010832Z_set`, full
  Set Transformer `20260531T_set_transformer_full`, factor robustness
  `20260531T_factor_robustness_macro_nonfred`, and claim ledger
  `20260531T_claim_ledger_macro_nonfred`.
- Model-comparison rows: `8`.
- Factor-alpha and cost rows: `7`.
- Review failures: `0`.
- Claim-validation failures: `0`.
- Reports generated as ignored private Markdown/LaTeX artifacts under `runs/`;
  table CSVs generated under ignored `artifacts/tables/`.

Latest private full-catalog publication-style diagnostic tables:

- Run: `20260531T_publication_tables_full_catalog_v2`.
- Inputs: full-catalog LightGBM `20260531T_lgbm_full_catalog`, factor
  robustness `20260531T_factor_robustness_full_catalog`, and claim ledger
  `20260531T_claim_ledger_full_catalog_v2`.
- Model-comparison rows: `8`.
- Factor-alpha and cost rows: `7`.
- Review failures: `0`.
- Claim-validation failures: `0`.

Latest private revision-safe FRED publication-style diagnostic tables:

- Run: `20260531T_publication_tables_fred_initial_release_v2`.
- Inputs: FRED initial-release LightGBM
  `20260531T_lgbm_fred_initial_release`, factor robustness
  `20260531T_factor_robustness_fred_initial_release`, and claim ledger
  `20260531T_claim_ledger_fred_initial_release_v2`.
- Model-comparison rows: `8`.
- Factor-alpha and cost rows: `7`.
- Review failures: `0`.
- Claim-validation failures: `0`.

Latest private visual pack:

- Run: `20260531T_visual_pack_macro_nonfred_v2`.
- Figure count: `7`.
- Model comparison rows: `8`.
- Firm explorer rows: `30`.
- Sector-geography matrix shape: `10 x 10`.
- Dashboard: ignored private HTML artifact under `artifacts/figures_html/`.
- Model card: ignored private report under `runs/`.

Latest private full-catalog visual pack:

- Run: `20260531T_visual_pack_full_catalog_v2`.
- Figure count: `7`.
- Model comparison rows: `8`.
- Firm explorer rows: `30`.
- Sector-geography matrix shape: `10 x 10`.
- Model card states that the full FRED/BLS/BEA/EIA catalog is live with
  configured no-lookahead timing and that revision-safe wording should remain
  limited to the separate FRED initial-release chain.

Latest private revision-safe FRED visual pack:

- Run: `20260531T_visual_pack_fred_initial_release_v3`.
- Figure count: `7`.
- Model comparison rows: `8`.
- Firm explorer rows: `30`.
- Sector-geography matrix shape: `10 x 10`.
- Model card states that FRED initial-release diagnostics are revision-safe
  only for the included FRED series; full-catalog no-lookahead diagnostics are
  tracked in `20260531T_visual_pack_full_catalog_v2`.

Latest private holdout protocol:

- Run: `20260531T_holdout_protocol_freeze`.
- Status: `frozen`.
- Holdout start: `2026-01-01`.
- Holdout opened: `false`.
- Selected development-sample model:
  `lgbm:20260531T_lgbm_macro_nonfred_v3:all_plus_macro`.
- Freeze checks: `13` passed, `0` failed.
- Development diagnostics recorded at freeze time: mean Q5-Q1 `0.010795`
  and factor-robust net Q5-Q1 `0.010245`.
- This is a protocol/model freeze only. It does not evaluate 2026 returns.

Latest public release-prep check:

- Hard-coded private Amarel path and allocation id removed from public runner
  scripts and docs.
- Runners require `SMB_PROJECT_ROOT` and `SMB_SLURM_JOB_ID`.
- CI now runs the public safety scan, release audit, and unit tests.
- Local checks passed: release audit, public safety scan, and 72 unit tests.
- Allocation-backed checks passed on the active compute environment: release
  audit, public safety scan, and 72 unit tests.
- Private manifest frontier audits `20260531T_private_state_audit` locally and
  `20260531T_private_state_audit_remote` on allocation `5752806` passed with
  `120` checks passed, zero failures, and zero blockers.
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
- Official non-FRED macro execution completed in
  `20260531T_macro_nonfred_full`; the full FRED/BLS/BEA/EIA configured-lag
  catalog completed in `20260531T_macro_full_catalog_delayed`.
- FRED initial-release/realtime support is implemented through
  `timing: fred_initial_release`, `fred_vintage_all`, and
  `fred_vintage_changes`. The limited live run
  `20260531T_fred_initial_release_guarded` executed the example catalog with
  true realtime availability dates.

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

- Keep revision-safe wording limited to the included FRED initial-release
  series; the full configured-lag catalog is no-lookahead but not true
  historical-vintage evidence.
- Interpret the LightGBM ablation carefully: segment-only features rank returns
  well, while the non-FRED `all_plus_macro` variant currently produces the
  strongest long-short spread and factor-alpha diagnostics.
- Interpret factor robustness carefully: no-return-or-market and
  no-market-factor LightGBM variants were strongest before macro integration;
  after the non-FRED macro tensor run, `all_plus_macro` is the strongest
  diagnostic. The publication-style tables remain diagnostic and are not final
  paper claims.
- Interpret the Deep Sets benchmark carefully: the simple set-only encoder has
  positive rank IC but weak long-short spread, and the first controls variant
  does not improve rank IC.
- Interpret the full Set Transformer carefully: it runs on CUDA but does not
  improve the economic spread diagnostics in the current development sample.
- Push only after explicit user approval and a final clean audit.
