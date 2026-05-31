# Release Audit

Public release gate:

- Private credentials are represented only by empty keys in `.env.example`.
- Real `.env` files, raw data, run logs, generated tables, generated figures,
  dashboards, caches, and package metadata are ignored.
- Amarel runner scripts require `SMB_PROJECT_ROOT` and `SMB_SLURM_JOB_ID` from
  the caller's environment instead of hard-coding a private cluster path or job
  id.
- Public tests use synthetic fixtures or code-level checks only.
- `docs/completion_audit.md` records verified, data-gated, and not-yet-final
  requirements before any push.
- No GitHub remote is required for this preparation step, and nothing should be
  pushed until the user explicitly asks.

Verification commands:

```bash
python scripts/public_safety_scan.py
python scripts/release_audit.py
python -m unittest discover -s tests
```

The release audit is intentionally conservative. It fails if private data
folders are tracked, required release files are missing, private operational
paths appear in public text files, or core ignore rules are absent.
