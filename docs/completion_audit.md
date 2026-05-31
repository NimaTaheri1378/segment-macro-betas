# Completion Audit

This audit tracks the original execution plan against current repository state.
It is intentionally conservative: code paths, private run manifests, tests, and
release checks are treated as evidence only for the scope they actually cover.

## Verified Complete

| Requirement | Evidence |
|---|---|
| Work stays inside the approved project tree | Public runners require `SMB_PROJECT_ROOT`; path guards reject outputs outside the project root. |
| Private data and generated outputs stay untracked | `.gitignore`, `scripts/public_safety_scan.py`, and `scripts/release_audit.py`; current `git status` shows only ignored private folders. |
| WRDS schema audit freezes table contracts | `configs/schema_map.yml`; private schema run `20260530T222000Z`. |
| Full 2006-2025 WRDS development extract exists | Private extract run `20260530T233446Z`; row counts summarized in `docs/status.md`. |
| Filing-date activation is implemented and validated | `src/segment_macro_betas/filing_dates.py`, panel integration, and private panel run `20260531T003936Z_panel_filing` with zero activation violations. |
| Baseline portfolio and cross-sectional diagnostics are implemented | `src/segment_macro_betas/baselines.py`; filing-date baseline run `20260531T004841Z_baseline_filing`. |
| LightGBM expanding-window ablations are implemented | `src/segment_macro_betas/lgbm_benchmark.py`; filing-date ablation run `20260531T005001Z_lgbm_filing`; macro-aware run `20260531T_lgbm_macro_nonfred_v3`. |
| Segment-set model extension is implemented | `src/segment_macro_betas/segment_set_model.py`; full Deep Sets run `20260531T010832Z_set`; full CUDA Set Transformer run `20260531T_set_transformer_full`. |
| Official non-FRED macro API execution is implemented and executed | `src/segment_macro_betas/macro_engine.py`; private BLS/BEA/EIA run `20260531T_macro_nonfred_full` with 2,480 rows and `lookahead_safe=true`. |
| Firm-geography-macro tensor is implemented and executed | `src/segment_macro_betas/macro_tensor.py`; private tensor run `20260531T_macro_tensor_nonfred_v2` with 936,897 panel rows, 2,811,167 joined token rows, 9 macro features, and 100% macro coverage. |
| Factor-alpha and turnover robustness code is implemented | `src/segment_macro_betas/factor_robustness.py`; macro-aware private diagnostic run `20260531T_factor_robustness_macro_nonfred`. |
| Claim ledger and wording guardrails are implemented | `src/segment_macro_betas/claim_ledger.py`, `docs/claim_guardrails.md`, and macro-aware private claim-ledger run `20260531T_claim_ledger_macro_nonfred`. |
| Publication-style diagnostic tables are implemented | `src/segment_macro_betas/publication_tables.py`, `scripts/run_publication_tables.sh`, and macro-aware private run `20260531T_publication_tables_macro_nonfred` with zero review failures. |
| Visual pack and model card are implemented | `src/segment_macro_betas/visual_pack.py`; macro-aware visual pack run `20260531T_visual_pack_macro_nonfred_v2`. |
| 2026 holdout protocol is frozen without opening holdout data | `scripts/freeze_holdout_protocol.py`; private holdout freeze run `20260531T_holdout_protocol_freeze` selects `lgbm:all_plus_macro` from development diagnostics and records `holdout_opened=false`. |
| Public-safe GitHub preparation is implemented | `.github/workflows/ci.yml` runs public safety scan, release audit, and tests; release docs/checklist exist; private-state audit `20260531T_private_state_audit` has zero failures; no configured remote; no push. |

## Code-Complete But Data-Gated

| Requirement | Current State | Gate |
|---|---|---|
| Full FRED-inclusive macro catalog | Execute mode started the full catalog in `20260531T_macro_full_catalog_guarded_smoke`, completed two FRED series, and then stopped safely on HTTP `429` for `UNRATE`. | Wait before retrying; do not spam the FRED API. |
| Fully revision-safe macro interactions | FRED realtime/initial-release catalog support is implemented and dry-run verified in `20260531T_fred_initial_release_dry`; the current non-FRED macro tensor is no-lookahead but `revision_safe=false`. | Requires executing a true realtime/vintage macro source after FRED rate-limit cooldown before final revision-safe claims. |

## Not Yet Final-Claim Ready

| Item | Reason |
|---|---|
| 2026 held-out evaluation | The model/protocol freeze is recorded, but `holdout_opened=false`; no 2026 performance has been evaluated. |
| Final transaction-cost and alpha claims | Macro-aware publication-style diagnostic tables now exist, but final claims still require revision-safe macro evidence, future holdout-result review, and author sign-off on manuscript wording. |
| Final paper claims | Current numbers are diagnostics with guardrails; model-card and docs avoid final asset-pricing claims. |

## Latest Verification

Local checks:

```bash
python scripts/release_audit.py
python scripts/public_safety_scan.py
python scripts/private_state_audit.py --run-id 20260531T_private_state_audit
python -m unittest discover -s tests
git diff --check
```

Allocation-backed checks:

```bash
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 python scripts/release_audit.py
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 python scripts/public_safety_scan.py
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 env PYTHONPATH=src python -m unittest discover -s tests
```

The current release state is suitable for a public-safe code push only after
the user explicitly authorizes creating/configuring the GitHub remote and
pushing. It is not a final empirical paper package until the gated FRED,
revision-safe macro, and 2026 holdout items are resolved.
