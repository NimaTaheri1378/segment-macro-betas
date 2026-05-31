# Reproducibility

Private WRDS data are never committed. Every private run writes:

- A manifest with query text, row counts, output paths, and validation checks.
- Compact logs under `runs/<run_id>/logs/`.
- Private Parquet outputs under `data/interim/<run_id>/`.

Large pulls should be sharded by natural date units and rerun from completed
shards rather than restarting from scratch.
