# Segment Macro Betas

Point-in-time equity asset-pricing study of segment-implied macro exposure.

The project turns Compustat geographic segment disclosures into firm-level
macro exposure measures, links those measures to CRSP returns through CCM, and
tests whether disclosure-implied macro betas predict cross-sectional returns.

Current implementation status:

- WRDS schema audit completed and frozen in `configs/schema_map.yml`.
- Smoke-panel pipeline implemented for a tiny private WRDS sample.
- Full annual WRDS shard extraction, monthly panel construction, first-pass
  baselines, and an expanding-window LightGBM benchmark are implemented.
- Official non-FRED macro extraction, macro tensor construction, and
  macro-aware LightGBM diagnostics are implemented for private runs.
- Public repository files are code, configs, docs, and tests only.
- Private data, credentials, and run logs are intentionally excluded from
  version control.

Run public-safe checks locally:

```bash
python scripts/public_safety_scan.py
python scripts/release_audit.py
python -m unittest discover -s tests
```

On Amarel, set the approved project root and allocation id in your shell before
calling any runner:

```bash
export SMB_PROJECT_ROOT="/path/to/Segment Macro Betas"
export SMB_SLURM_JOB_ID="<approved-allocation-id>"
```

Run the WRDS smoke panel on Amarel only inside the approved Slurm allocation:

```bash
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_smoke_panel.sh
```

Plan or execute the sharded full extraction:

```bash
# Dry-run query contracts only
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_full_extract.sh

# Full annual shards, launched only after review
EXECUTE=1 YEARS=2006-2025 INCLUDE_DAILY=0 \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_full_extract.sh
```

Plan or execute the macro engine:

```bash
EXECUTE=0 srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_macro_engine.sh
```

The macro catalog lives in `configs/macro_series.yml` and supports FRED, BLS,
BEA, and EIA rows. Live execution requires an untracked compute-host `.env`;
manifests record credential presence and timing flags, never credential values.

Build the firm-geography-macro tensor from cached macro data:

```bash
RAW_RUN_ID=20260530T233446Z \
PANEL_RUN_ID=20260531T003936Z_panel_filing \
MACRO_RUN_ID=<macro-run-id> \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=2 bash scripts/run_macro_tensor.sh
```

Build the monthly modeling panel from a completed raw run:

```bash
RAW_RUN_ID=20260530T233446Z \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_build_panel.sh
```

Add the targeted Compustat filing-date supplement before rebuilding the panel:

```bash
RAW_RUN_ID=20260530T233446Z EXECUTE=1 \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_filing_dates_extract.sh
```

Run first-pass portfolio and cross-sectional baselines:

```bash
PANEL_RUN_ID=20260531T003936Z_panel_filing \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_baselines.sh
```

Run the LightGBM expanding-window benchmark:

```bash
PANEL_RUN_ID=20260531T003936Z_panel_filing \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=8 bash scripts/run_lgbm_benchmark.sh
```

By default this runner executes feature ablations: `all`,
`no_market_factors`, `no_return_or_market`, `segment_only`, and
`non_segment_controls`. Override with `VARIANTS=segment_only` for a targeted
run. `LGBM_DEVICE_TYPE=auto` attempts LightGBM GPU training first and records
any CPU fallback in the manifest.

For macro-aware diagnostics, pass a macro tensor run and dataset name:

```bash
PANEL_RUN_ID=<macro-tensor-run-id> \
PANEL_DATASET=macro_tensor_panel \
VARIANTS=macro_only,segment_plus_macro,all_plus_macro,segment_only,non_segment_controls \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=8 bash scripts/run_lgbm_benchmark.sh
```

`macro_only` is expected to be diagnostic-only when the macro inputs are global
monthly states with no within-month cross-sectional variation.

Run the Deep Sets segment-set extension:

```bash
RAW_RUN_ID=20260530T233446Z PANEL_RUN_ID=20260531T003936Z_panel_filing \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=8 bash scripts/run_segment_set_model.sh
```

`SET_DEVICE_TYPE=auto` uses CUDA for the PyTorch Deep Sets model whenever the
allocation exposes a GPU.
Use `VARIANTS=set_transformer` for the optional self-attention segment-set
encoder.

Run factor-alpha and turnover robustness from cached predictions:

```bash
PANEL_RUN_ID=20260531T003936Z_panel_filing \
MODEL_RUNS=lgbm:20260531T_lgbm_macro_nonfred_v3,deepsets:20260531T010832Z_set,deepsets:20260531T_set_transformer_full \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=2 bash scripts/run_factor_robustness.sh
```

Generate the private claim ledger and table inventory from cached diagnostics:

```bash
PANEL_RUN_ID=20260531T003936Z_panel_filing \
LGBM_RUN_ID=20260531T_lgbm_macro_nonfred_v3 \
SET_RUN_ID=20260531T010832Z_set \
SET_RUN_IDS=20260531T010832Z_set,20260531T_set_transformer_full \
FACTOR_RUN_ID=20260531T_factor_robustness_macro_nonfred \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_claim_ledger.sh
```

Render private publication-style diagnostic tables after claim-ledger review:

```bash
PANEL_RUN_ID=20260531T003936Z_panel_filing \
LGBM_RUN_ID=20260531T_lgbm_macro_nonfred_v3 \
SET_RUN_ID=20260531T010832Z_set \
SET_RUN_IDS=20260531T010832Z_set,20260531T_set_transformer_full \
FACTOR_RUN_ID=20260531T_factor_robustness_macro_nonfred \
CLAIM_RUN_ID=20260531T_claim_ledger_macro_nonfred \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=1 bash scripts/run_publication_tables.sh
```

Generate the private visual pack and model card:

```bash
RAW_RUN_ID=20260530T233446Z \
PANEL_RUN_ID=20260531T003936Z_panel_filing \
BASELINE_RUN_ID=20260531T004841Z_baseline_filing \
LGBM_RUN_ID=20260531T_lgbm_macro_nonfred_v3 \
SET_RUN_ID=20260531T010832Z_set \
SET_RUN_IDS=20260531T010832Z_set,20260531T_set_transformer_full \
srun --overlap --jobid="$SMB_SLURM_JOB_ID" --chdir="$SMB_PROJECT_ROOT" \
  --ntasks=1 --cpus-per-task=4 bash scripts/run_visual_pack.sh
```
