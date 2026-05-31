from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REQUIRED_PUBLIC_FILES = [
    ".env.example",
    ".github/workflows/ci.yml",
    ".gitignore",
    "CITATION.cff",
    "DATA_ACCESS.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "configs/data.yml",
    "configs/features.yml",
    "configs/macro_series.yml",
    "configs/macro_series_fred_initial_release.example.yml",
    "configs/model_lgbm.yml",
    "configs/model_set.yml",
    "configs/pipeline.yml",
    "configs/schema_map.yml",
    "docs/claim_guardrails.md",
    "docs/completion_audit.md",
    "docs/data_contract.md",
    "docs/github_release_checklist.md",
    "docs/index.md",
    "docs/methodology.md",
    "docs/figures/diagnostic_snapshot.svg",
    "docs/figures/pipeline_architecture.svg",
    "docs/figures/release_boundary.svg",
    "docs/output_inventory.md",
    "docs/release_audit.md",
    "docs/release_notes.md",
    "docs/reproducibility.md",
    "docs/status.md",
    "scripts/_amarel_env.sh",
    "scripts/freeze_holdout_protocol.py",
    "scripts/private_state_audit.py",
    "scripts/public_safety_scan.py",
    "scripts/run_claim_ledger.sh",
    "scripts/run_baselines.sh",
    "scripts/run_build_panel.sh",
    "scripts/run_factor_robustness.sh",
    "scripts/run_filing_dates_extract.sh",
    "scripts/run_full_extract.sh",
    "scripts/run_lgbm_benchmark.sh",
    "scripts/run_macro_engine.sh",
    "scripts/run_macro_tensor.sh",
    "scripts/run_publication_tables.sh",
    "scripts/run_schema_audit.sh",
    "scripts/run_segment_set_model.sh",
    "scripts/run_smoke_figures.sh",
    "scripts/run_smoke_panel.sh",
    "scripts/run_visual_pack.sh",
    "scripts/schema_audit.py",
    "src/segment_macro_betas/baselines.py",
    "src/segment_macro_betas/claim_ledger.py",
    "src/segment_macro_betas/factor_robustness.py",
    "src/segment_macro_betas/filing_dates.py",
    "src/segment_macro_betas/full_extract.py",
    "src/segment_macro_betas/lgbm_benchmark.py",
    "src/segment_macro_betas/macro_engine.py",
    "src/segment_macro_betas/macro_tensor.py",
    "src/segment_macro_betas/panel_builder.py",
    "src/segment_macro_betas/publication_tables.py",
    "src/segment_macro_betas/segment_set_model.py",
    "src/segment_macro_betas/smoke_panel.py",
    "src/segment_macro_betas/visual_pack.py",
    "src/segment_macro_betas/visualise/smoke_figures.py",
    "src/segment_macro_betas/wrds_access.py",
]
PRIVATE_DATA_LIKE_SUFFIXES = (
    ".arrow",
    ".csv",
    ".db",
    ".dta",
    ".duckdb",
    ".feather",
    ".gif",
    ".h5",
    ".hdf",
    ".html",
    ".jpeg",
    ".jpg",
    ".joblib",
    ".parquet",
    ".pkl",
    ".png",
    ".rds",
    ".sas7bdat",
    ".sqlite",
    ".tsv",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
)
PRIVATE_TRACKED_PREFIXES = (
    ".codex/",
    "artifacts/",
    "data/",
    "runs/",
)
PRIVATE_TRACKED_NAMES = {
    ".env",
}
PRIVATE_MARKERS = [
    re.compile(r"/(?:scratch|home)/[A-Za-z0-9_.-]+/"),
    re.compile(r"(?i)PROJECT_ROOT=\"/(?:scratch|home)/"),
    re.compile(r"(?i)EXPECTED_JOB_ID=\"\d+\""),
    re.compile(r"(?i)--jobid=\d+"),
]


def git_lines(args: list[str]) -> list[str] | None:
    try:
        proc = subprocess.run(["git", *args], check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def iter_public_text_files(root: Path):
    skip_dirs = {".git", ".codex", "artifacts", "data", "runs", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs or part.endswith(".egg-info") for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        yield path, text


def audit(root: Path) -> list[str]:
    findings: list[str] = []

    for rel in REQUIRED_PUBLIC_FILES:
        if not (root / rel).exists():
            findings.append(f"missing required public file: {rel}")

    tracked = git_lines(["ls-files"])
    if tracked is not None:
        for rel in tracked:
            if rel in PRIVATE_TRACKED_NAMES or rel.startswith(PRIVATE_TRACKED_PREFIXES):
                findings.append(f"private path is tracked: {rel}")
            if rel.lower().endswith(PRIVATE_DATA_LIKE_SUFFIXES):
                findings.append(f"private/binary data-like file is tracked: {rel}")

    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    for pattern in [".env", ".env.*", "data/", "runs/", "artifacts/tables/", "artifacts/figures_static/", "artifacts/figures_html/", "*.egg-info/"]:
        if pattern not in gitignore:
            findings.append(f".gitignore missing private pattern: {pattern}")

    for path, text in iter_public_text_files(root):
        rel = path.relative_to(root).as_posix()
        for pattern in PRIVATE_MARKERS:
            if pattern.search(text):
                findings.append(f"private operational marker in public file: {rel}")
                break

    return findings


def main() -> int:
    root = Path.cwd()
    findings = audit(root)
    if findings:
        print("release_audit_failed")
        for finding in findings:
            print(f"  {finding}")
        return 1
    print("release_audit_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
