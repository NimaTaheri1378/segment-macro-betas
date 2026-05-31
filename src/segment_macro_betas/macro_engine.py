from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yaml

from segment_macro_betas.io_utils import atomic_write_json, atomic_write_parquet, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


DEFAULT_MACRO_SERIES = [
    {
        "source": "fred",
        "series_id": "FEDFUNDS",
        "series_name": "federal_funds_rate",
        "macro_area": "GLOBAL",
        "release_lag_days": 7,
    },
    {
        "source": "fred",
        "series_id": "CPIAUCSL",
        "series_name": "us_cpi",
        "macro_area": "GLOBAL",
        "release_lag_days": 21,
    },
    {
        "source": "fred",
        "series_id": "UNRATE",
        "series_name": "us_unemployment",
        "macro_area": "GLOBAL",
        "release_lag_days": 7,
    },
    {
        "source": "fred",
        "series_id": "INDPRO",
        "series_name": "us_industrial_production",
        "macro_area": "GLOBAL",
        "release_lag_days": 21,
    },
    {
        "source": "fred",
        "series_id": "DTWEXBGS",
        "series_name": "trade_weighted_usd",
        "macro_area": "GLOBAL",
        "release_lag_days": 3,
    },
]
DEFAULT_FRED_SERIES = {item["series_id"]: item["series_name"] for item in DEFAULT_MACRO_SERIES}


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


def normalise_series_config(raw: dict[str, Any]) -> dict[str, Any]:
    source = str(raw.get("source", "fred")).strip().lower()
    series_id = str(raw.get("series_id", "")).strip().upper()
    if not series_id:
        raise ValueError("Macro series config is missing series_id.")
    if source != "fred":
        raise ValueError(f"Unsupported macro source {source!r} for {series_id}.")
    release_lag_days = int(raw.get("release_lag_days", 0))
    return {
        "source": source,
        "series_id": series_id,
        "series_name": str(raw.get("series_name") or DEFAULT_FRED_SERIES.get(series_id) or series_id.lower()).strip(),
        "macro_area": str(raw.get("macro_area", "GLOBAL")).strip().upper(),
        "release_lag_days": release_lag_days,
        "frequency": str(raw.get("frequency", "monthly")).strip().lower(),
        "timing": str(raw.get("timing", "configured_release_lag")).strip().lower(),
        "revision_safe": bool(raw.get("revision_safe", False)),
    }


def load_series_catalog(project_root: Path, config_path: Path | None = None) -> list[dict[str, Any]]:
    path = config_path or project_root / "configs" / "macro_series.yml"
    if not path.exists():
        return [normalise_series_config(item) for item in DEFAULT_MACRO_SERIES]
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("series", data if isinstance(data, list) else [])
    if not isinstance(rows, list):
        raise ValueError(f"Macro series config must contain a list under 'series': {path}")
    catalog = [normalise_series_config(item) for item in rows]
    if not catalog:
        raise ValueError(f"Macro series config is empty: {path}")
    return catalog


def add_configured_availability(df: pd.DataFrame, series: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    observation_date = pd.to_datetime(out["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    lag = pd.to_timedelta(int(series.get("release_lag_days", 0)), unit="D")
    out["available_date"] = observation_date + lag
    out["timing_source"] = series.get("timing", "configured_release_lag")
    out["revision_safe"] = bool(series.get("revision_safe", False))
    out["lookahead_safe"] = True
    return out


def fred_observations(series: dict[str, Any], api_key: str, start: str, end: str) -> pd.DataFrame:
    series_id = series["series_id"]
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
    df["series_name"] = series["series_name"]
    df["macro_area"] = series["macro_area"]
    df["source"] = series["source"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df = add_configured_availability(df, series)
    keep = [
        "source",
        "series_id",
        "series_name",
        "macro_area",
        "date",
        "available_date",
        "value",
        "realtime_start",
        "realtime_end",
        "timing_source",
        "lookahead_safe",
        "revision_safe",
    ]
    return df[keep]


def run_macro(project_root: Path, run_id: str, start: str, end: str, execute: bool, series_config: Path | None = None) -> dict:
    paths = make_run_paths(project_root, run_id)
    env_values = {**read_env_file(project_root / ".env"), **os.environ}
    catalog = load_series_catalog(project_root, series_config)
    manifest = {
        "run_id": run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "start": start,
        "end": end,
        "execute": execute,
        "series": [{key: item[key] for key in ("source", "series_id", "series_name", "macro_area", "release_lag_days", "timing", "revision_safe")} for item in catalog],
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
    for series in catalog:
        frames.append(fred_observations(series, api_key, start, end))
    macro = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path = ensure_within(project_root, project_root / "data" / "raw" / run_id / "macro_fred_monthly.parquet")
    atomic_write_parquet(macro, out_path)
    manifest["outputs"]["fred_monthly"] = str(out_path)
    manifest["rows"] = int(len(macro))
    manifest["checks"] = {
        "series_count": int(macro["series_id"].nunique()) if len(macro) and "series_id" in macro.columns else 0,
        "lookahead_safe": bool(macro["lookahead_safe"].all()) if len(macro) and "lookahead_safe" in macro.columns else False,
        "revision_safe": bool(macro["revision_safe"].all()) if len(macro) and "revision_safe" in macro.columns else False,
        "available_date_min": str(pd.to_datetime(macro["available_date"]).min().date()) if len(macro) else None,
        "available_date_max": str(pd.to_datetime(macro["available_date"]).max().date()) if len(macro) else None,
    }
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
            f"- Series count: `{manifest.get('checks', {}).get('series_count', '')}`",
            f"- Lookahead-safe dates: `{manifest.get('checks', {}).get('lookahead_safe', '')}`",
            f"- Revision-safe vintages: `{manifest.get('checks', {}).get('revision_safe', '')}`",
            "",
            "Credential values are not stored in this manifest.",
            "Configured release lags are no-lookahead timing controls, not proof of unrevised vintage data.",
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
    parser.add_argument("--series-config", default=None)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    series_config = Path(args.series_config) if args.series_config else None
    if series_config and not series_config.is_absolute():
        series_config = project_root / series_config
    manifest = run_macro(project_root, args.run_id, args.start, args.end, args.execute, series_config)
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'macro_engine.json'}")
    return 0 if manifest["status"] in {"ok", "dry_run_ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
