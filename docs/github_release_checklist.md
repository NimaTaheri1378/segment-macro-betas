# GitHub Release Checklist

This checklist is for preparing the public repository. It does not authorize a
push by itself.

Before creating or attaching a remote:

- Confirm `git remote -v` is empty unless the user has explicitly approved a
  target.
- Run `python scripts/public_safety_scan.py`.
- Run `python scripts/release_audit.py`.
- Run `python -m unittest discover -s tests`.
- Confirm `git status --short --ignored` shows only tracked public code changes
  and ignored private folders.
- Confirm `.env`, raw WRDS extracts, run logs, generated private tables,
  generated figures, dashboards, and package metadata are ignored.
- Review `docs/completion_audit.md` and `docs/status.md` for claim discipline:
  private diagnostics are research status notes, not final paper claims.

Allowed to push after explicit approval:

- Source code under `src/`.
- Runner scripts that require caller-provided `SMB_PROJECT_ROOT` and
  `SMB_SLURM_JOB_ID`.
- Public configs, templates, tests, CI, and documentation.

Never push:

- Credential values or local `.env` files.
- WRDS extracts, intermediate panels, model predictions, private tables,
  generated figures, dashboards, or logs.
- Private Amarel paths, job ids, usernames, phone/MFA material, or passwords.
