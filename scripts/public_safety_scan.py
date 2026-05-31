from __future__ import annotations

import re
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"(?i)(fred|bls|bea|eia)_api_key[^\S\r\n]*=[^\S\r\n]*[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)sec_user_agent[^\S\r\n]*=[^\r\n]+@.+"),
    re.compile(r"[A-Fa-f0-9]{32,}"),
    re.compile(r"@gmail\.com", re.IGNORECASE),
]

SKIP_DIRS = {".codex", ".git", "runs", "data", "artifacts", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def iter_public_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    root = Path.cwd()
    findings: list[str] = []
    for path in iter_public_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(str(path))
                break
    if findings:
        print("Potential secret-like content found:")
        for finding in findings:
            print(f"  {finding}")
        return 1
    print("public_safety_scan_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
