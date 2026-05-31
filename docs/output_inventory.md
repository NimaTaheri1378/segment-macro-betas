# Output Inventory

This repository is prepared as a public-safe code and documentation release.
Aggregate schematics and reviewed summary figures may be tracked, while raw
WRDS-derived data, row-level outputs, private dashboards, and machine-readable
artifacts that could reconstruct licensed data remain ignored.

Tracked public assets:

- Source code for schema audit, WRDS extracts, panel construction, macro
  tensors, baselines, LightGBM, segment-set models, factor robustness, claim
  ledgers, publication tables, and visual packs.
- Slurm runner scripts under `scripts/` that require caller-provided project
  root and allocation id through `SMB_PROJECT_ROOT` and `SMB_SLURM_JOB_ID`.
- Public configs, tests, documentation, CI, `.env.example`, and data-access
  notes.
- Public-safe SVG figures under `docs/figures/` that summarize the pipeline,
  diagnostic frontier, and release boundary without row-level data.

Private ignored outputs verified locally:

- `runs/20260531T_claim_ledger_full_catalog_v2/`: full-catalog claim ledger
  with zero validation failures and zero blocked claims.
- `runs/20260531T_publication_tables_full_catalog_v2/`: publication-style
  model comparison and factor/cost tables with zero review failures.
- `artifacts/figures_static/20260531T_visual_pack_full_catalog_v2/`: seven
  nonempty PNG figures generated from private artifacts.
- `artifacts/figures_html/20260531T_visual_pack_full_catalog_v2/dashboard.html`:
  nonempty private dashboard HTML that references the ignored static figures.
- `runs/20260531T_visual_pack_fred_initial_release_v3/` and matching ignored
  figure folders: revision-safe FRED initial-release visual diagnostics for the
  included FRED series.
- `runs/20260531T_holdout_protocol_freeze/`: frozen 2026 holdout protocol with
  the holdout unopened.

The private visual pack currently contains:

- Activation source coverage.
- Sample model coverage.
- Sector-geography matrix.
- Foreign-share distribution.
- Exposure time series.
- Model rank-IC comparison.
- Model long-short spread comparison.
- Dashboard HTML and model-card report.

Regeneration commands are public-safe and live in the tracked runner scripts:

```bash
bash scripts/run_schema_audit.sh <run_id>
bash scripts/run_full_extract.sh <run_id>
bash scripts/run_filing_dates_extract.sh <run_id>
bash scripts/run_build_panel.sh <run_id>
bash scripts/run_macro_engine.sh <run_id>
bash scripts/run_macro_tensor.sh <run_id>
bash scripts/run_lgbm_benchmark.sh <run_id>
bash scripts/run_segment_set_model.sh <run_id>
bash scripts/run_factor_robustness.sh <run_id>
bash scripts/run_claim_ledger.sh <run_id>
bash scripts/run_publication_tables.sh <run_id>
bash scripts/run_visual_pack.sh <run_id>
```

Before any push, run:

```bash
python scripts/public_safety_scan.py
python scripts/release_audit.py
python scripts/private_state_audit.py --run-id <run_id>
python -m unittest discover -s tests
```

The public release may include reviewed aggregate figures and tables. It should
not include row-level WRDS-derived outputs, private dashboards, caches, logs,
or any file that redistributes or reconstructs licensed data.
