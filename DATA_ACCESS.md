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
templates, synthetic fixtures, and public-safe manifests only. Tables and
figures generated from private WRDS data remain ignored until explicitly
reviewed for publication.

Set the active Amarel project workspace outside version control:

```text
SMB_PROJECT_ROOT=/path/to/Segment Macro Betas
SMB_SLURM_JOB_ID=<approved-allocation-id>
```

Compute work should run only through the user-approved Slurm allocation unless
the user explicitly changes the allocation.
