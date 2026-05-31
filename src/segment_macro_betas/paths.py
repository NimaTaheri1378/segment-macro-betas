from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SCRATCH_PROJECT_ROOT = Path("/scratch/nt612/Github/Segment Macro Betas")


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str
    run_root: Path
    logs: Path
    manifests: Path
    reports: Path
    data_interim: Path
    artifacts_tables: Path


def resolve_project_root(value: str | None = None) -> Path:
    raw = value or os.environ.get("SMB_PROJECT_ROOT") or os.getcwd()
    return Path(raw).expanduser().resolve()


def ensure_within(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing path outside project root: {resolved_path}") from exc
    return resolved_path


def require_project_root(root: Path, expected: Path | None = None) -> Path:
    resolved = root.resolve()
    if expected is not None and resolved != expected.resolve():
        raise ValueError(f"Unexpected project root {resolved}; expected {expected.resolve()}")
    return resolved


def make_run_paths(project_root: Path, run_id: str) -> RunPaths:
    root = project_root.resolve()
    run_root = ensure_within(root, root / "runs" / run_id)
    paths = RunPaths(
        root=root,
        run_id=run_id,
        run_root=run_root,
        logs=ensure_within(root, run_root / "logs"),
        manifests=ensure_within(root, run_root / "manifests"),
        reports=ensure_within(root, run_root / "reports"),
        data_interim=ensure_within(root, root / "data" / "interim" / run_id),
        artifacts_tables=ensure_within(root, root / "artifacts" / "tables" / run_id),
    )
    for path in (paths.logs, paths.manifests, paths.reports, paths.data_interim, paths.artifacts_tables):
        path.mkdir(parents=True, exist_ok=True)
    return paths
