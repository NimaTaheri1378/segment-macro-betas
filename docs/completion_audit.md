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
| LightGBM expanding-window ablations are implemented | `src/segment_macro_betas/lgbm_benchmark.py`; filing-date ablation run `20260531T005001Z_lgbm_filing`. |
| Segment-set model extension is implemented | `src/segment_macro_betas/segment_set_model.py`; full Deep Sets run `20260531T010832Z_set`. |
| Visual pack and model card are implemented | `src/segment_macro_betas/visual_pack.py`; visual pack run `20260531T_visual_pack`. |
| Public-safe GitHub preparation is implemented | `.github/workflows/ci.yml`, release docs, release audit, public safety scan, no configured remote, and no push. |

## Code-Complete But Data-Gated

| Requirement | Current State | Gate |
|---|---|---|
| Official macro API execution | `src/segment_macro_betas/macro_engine.py` can read untracked `.env` and execute FRED pulls; dry run `20260531T_release_macro_dry` passed. | Requires untracked compute-host secrets and explicit execute run. |
| Firm-geography-macro tensor | `src/segment_macro_betas/macro_tensor.py` and `scripts/run_macro_tensor.sh` build tensors from cached macro Parquet files. | Requires a private macro run with release or realtime availability dates. |
| Fully vintage-safe macro interactions | Tensor builder honors `available_date`, `realtime_start`, or `release_date` and flags fallback timing. | Requires cached macro data with vintage/release timing fields. |

## Not Yet Final-Claim Ready

| Item | Reason |
|---|---|
| 2026 held-out evaluation | The proposal explicitly keeps 2026 untouched; no final holdout run should happen until modeling choices are frozen. |
| Factor alphas and transaction-cost robustness | The current repo has diagnostics and model benchmarks, not final factor-alpha tables or net-cost portfolio evidence. |
| Set Transformer stretch model | Deep Sets is implemented; Set Transformer remains a stretch extension. |
| Final paper claims | Current numbers are diagnostics with guardrails; model-card and docs avoid final asset-pricing claims. |

## Latest Verification

Local checks:

```bash
python scripts/release_audit.py
python scripts/public_safety_scan.py
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
pushing. It is not a final empirical paper package until the gated macro and
robustness items above are resolved.
