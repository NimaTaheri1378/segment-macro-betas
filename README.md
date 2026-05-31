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
- Public repository files are code, configs, docs, and tests only.
- Private data, credentials, and run logs are intentionally excluded from
  version control.

Run public-safe checks locally:

```bash
python scripts/public_safety_scan.py
python -m unittest discover -s tests
```

Run the WRDS smoke panel on Amarel only inside the approved Slurm allocation:

```bash
srun --overlap --jobid=5752806 --chdir="/scratch/nt612/Github/Segment Macro Betas" \
  --ntasks=1 --cpus-per-task=1 scripts/run_smoke_panel.sh
```

Plan or execute the sharded full extraction:

```bash
# Dry-run query contracts only
srun --overlap --jobid=5752806 --chdir="/scratch/nt612/Github/Segment Macro Betas" \
  --ntasks=1 --cpus-per-task=1 scripts/run_full_extract.sh

# Full annual shards, launched only after review
EXECUTE=1 YEARS=2006-2025 INCLUDE_DAILY=0 \
srun --overlap --jobid=5752806 --chdir="/scratch/nt612/Github/Segment Macro Betas" \
  --ntasks=1 --cpus-per-task=1 scripts/run_full_extract.sh
```

Plan or execute the macro engine:

```bash
EXECUTE=0 srun --overlap --jobid=5752806 --chdir="/scratch/nt612/Github/Segment Macro Betas" \
  --ntasks=1 --cpus-per-task=1 scripts/run_macro_engine.sh
```

Build the monthly modeling panel from a completed raw run:

```bash
RAW_RUN_ID=20260530T233446Z \
srun --overlap --jobid=5752806 --chdir="/scratch/nt612/Github/Segment Macro Betas" \
  --ntasks=1 --cpus-per-task=1 scripts/run_build_panel.sh
```

Run first-pass portfolio and cross-sectional baselines:

```bash
PANEL_RUN_ID=20260530T234643Z \
srun --overlap --jobid=5752806 --chdir="/scratch/nt612/Github/Segment Macro Betas" \
  --ntasks=1 --cpus-per-task=1 scripts/run_baselines.sh
```

Run the LightGBM expanding-window benchmark:

```bash
PANEL_RUN_ID=20260530T234643Z \
srun --overlap --jobid=5752806 --chdir="/scratch/nt612/Github/Segment Macro Betas" \
  --ntasks=1 --cpus-per-task=8 scripts/run_lgbm_benchmark.sh
```

By default this runner executes feature ablations: `all`,
`no_market_factors`, `no_return_or_market`, `segment_only`, and
`non_segment_controls`. Override with `VARIANTS=segment_only` for a targeted
run.
