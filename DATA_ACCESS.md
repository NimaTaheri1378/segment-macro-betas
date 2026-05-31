# Data Access

This project is designed for private, entitlement-based data access. Raw WRDS
extracts, intermediate panels, credentials, and run logs must not be committed.

Required private sources:

- WRDS access for CRSP, CCM, Compustat North America, and Compustat Historical
  Segments.
- Local API credentials for official macro and energy sources when the macro
  engine is enabled.
- SEC EDGAR automated access should use a local User-Agent/contact string.

Public repository artifacts should contain code, documentation, configuration
templates, synthetic fixtures, public-safe manifests, and aggregate figures or
tables that have been explicitly reviewed for publication. Raw data,
row-level outputs, firm/security-month panels, private dashboards, and
machine-readable files that could reconstruct licensed data remain excluded.

Publications, presentations, and public repositories that rely on WRDS data
should acknowledge WRDS and the relevant third-party data suppliers, consistent
with institutional subscription terms. This repository does not redistribute
raw or row-level WRDS data.

Set the active Amarel project workspace outside version control:

```text
SMB_PROJECT_ROOT=/path/to/Segment Macro Betas
SMB_SLURM_JOB_ID=<approved-allocation-id>
```

For macro execution, copy `.env.example` to an untracked `.env` on the compute
host and fill only the variables needed for the run. The macro engine reads the
public FRED/BLS/BEA/EIA catalog in `configs/macro_series.yml`, writes timing
flags and row counts to manifests, and never writes credential values to
manifests or reports.

Compute work should run only through the user-approved Slurm allocation unless
the user explicitly changes the allocation.
