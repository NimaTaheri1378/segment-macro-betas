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
    "README.md",
    "pyproject.toml",
    "configs/data.yml",
    "configs/macro_series.yml",
    "configs/schema_map.yml",
    "docs/completion_audit.md",
    "docs/github_release_checklist.md",
    "docs/release_audit.md",
    "docs/release_notes.md",
    "docs/reproducibility.md",
    "docs/status.md",
    "scripts/_amarel_env.sh",
    "scripts/public_safety_scan.py",
    "scripts/run_factor_robustness.sh",
    "scripts/run_macro_engine.sh",
    "scripts/run_macro_tensor.sh",
]
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
            if rel.endswith((".parquet", ".pkl", ".zip")):
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
