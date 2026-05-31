from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from segment_macro_betas.io_utils import atomic_write_json, atomic_write_parquet, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


DEFAULT_FRED_SERIES = {
    "FEDFUNDS": "federal_funds_rate",
    "CPIAUCSL": "us_cpi",
    "UNRATE": "us_unemployment",
    "INDPRO": "us_industrial_production",
    "DTWEXBGS": "trade_weighted_usd",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def fred_observations(series_id: str, api_key: str, start: str, end: str) -> pd.DataFrame:
    query = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{query}"
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("observations", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "value", "series_id"])
    df["series_id"] = series_id
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    return df[["series_id", "date", "value", "realtime_start", "realtime_end"]]


def run_macro(project_root: Path, run_id: str, start: str, end: str, execute: bool) -> dict:
    paths = make_run_paths(project_root, run_id)
    env_values = {**read_env_file(project_root / ".env"), **os.environ}
    manifest = {
        "run_id": run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "start": start,
        "end": end,
        "execute": execute,
        "series": DEFAULT_FRED_SERIES,
        "outputs": {},
        "status": "planned",
        "credential_presence": {
            "FRED_API_KEY": bool(env_values.get("FRED_API_KEY")),
            "BLS_API_KEY": bool(env_values.get("BLS_API_KEY")),
            "BEA_API_KEY": bool(env_values.get("BEA_API_KEY")),
            "EIA_API_KEY": bool(env_values.get("EIA_API_KEY")),
            "SEC_USER_AGENT": bool(env_values.get("SEC_USER_AGENT")),
        },
    }
    if not execute:
        manifest["status"] = "dry_run_ok"
        atomic_write_json(paths.manifests / "macro_engine.json", manifest)
        atomic_write_text(paths.reports / "macro_engine_report.md", report_text(manifest))
        return manifest

    api_key = env_values.get("FRED_API_KEY")
    if not api_key:
        manifest["status"] = "missing_fred_api_key"
        atomic_write_json(paths.manifests / "macro_engine.json", manifest)
        atomic_write_text(paths.reports / "macro_engine_report.md", report_text(manifest))
        return manifest

    frames = []
    for series_id in DEFAULT_FRED_SERIES:
        frames.append(fred_observations(series_id, api_key, start, end))
    macro = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = ensure_within(project_root, project_root / "data" / "raw" / run_id / "macro_fred_monthly.parquet")
    atomic_write_parquet(macro, out_path)
    manifest["outputs"]["fred_monthly"] = str(out_path)
    manifest["rows"] = int(len(macro))
    manifest["status"] = "ok"
    atomic_write_json(paths.manifests / "macro_engine.json", manifest)
    atomic_write_text(paths.reports / "macro_engine_report.md", report_text(manifest))
    return manifest


def report_text(manifest: dict) -> str:
    return "\n".join(
        [
            "# Macro Engine Report",
            "",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Status: `{manifest['status']}`",
            f"- Window: `{manifest['start']}` to `{manifest['end']}`",
            f"- Execute: `{manifest['execute']}`",
            f"- Rows: `{manifest.get('rows', '')}`",
            "",
            "Credential values are not stored in this manifest.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_macro(project_root, args.run_id, args.start, args.end, args.execute)
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'macro_engine.json'}")
    return 0 if manifest["status"] in {"ok", "dry_run_ok", "missing_fred_api_key"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
