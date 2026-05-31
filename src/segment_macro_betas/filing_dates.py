from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from segment_macro_betas.config import load_schema_roles
from segment_macro_betas.io_utils import atomic_write_json, atomic_write_parquet, atomic_write_text, parquet_info
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root
from segment_macro_betas.wrds_access import connect_wrds, pgpass_metadata, qliteral, query_hash, table_ref


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def years_from_args(years: str | None, start: int, end: int) -> list[int]:
    if not years:
        return list(range(start, end + 1))
    result: list[int] = []
    for piece in years.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(piece))
    return sorted(set(result))


def filing_date_sql(schema_roles: dict[str, dict[str, str]], year: int) -> str:
    start = f"{year - 2}-01-01"
    end = f"{year}-12-31"
    funda = schema_roles["compustat_fundamentals_annual"]
    return f"""
select
    gvkey,
    datadate,
    fyear,
    fyr,
    fdate,
    pdate
from {table_ref(funda['library'], funda['table'])}
where datadate between {qliteral(start)} and {qliteral(end)}
  and indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
""".strip()


def prepare_filing_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gvkey"] = out["gvkey"].astype(str)
    for col in ("datadate", "fdate", "pdate"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    out = out[out["datadate"].notna()].copy()
    if "fdate" not in out.columns:
        out["fdate"] = pd.NaT
    if "pdate" not in out.columns:
        out["pdate"] = pd.NaT
    out["filing_date"] = out["pdate"].where(out["pdate"].notna(), out["fdate"])
    out["filing_date_source"] = "missing"
    out.loc[out["pdate"].notna(), "filing_date_source"] = "pdate"
    out.loc[out["pdate"].isna() & out["fdate"].notna(), "filing_date_source"] = "fdate"
    out = out.sort_values(["gvkey", "datadate", "filing_date"]).drop_duplicates(["gvkey", "datadate"], keep="first")
    keep = ["gvkey", "datadate", "fyear", "fyr", "fdate", "pdate", "filing_date", "filing_date_source"]
    return out[[col for col in keep if col in out.columns]]


def run_extract(project_root: Path, raw_run_id: str, run_id: str, years: list[int], execute: bool) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    schema_roles = load_schema_roles(project_root / "configs" / "schema_map.yml")
    data_root = ensure_within(project_root, project_root / "data" / "raw" / raw_run_id)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "raw_run_id": raw_run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "years": years,
        "execute": execute,
        "pgpass": pgpass_metadata(),
        "queries": [],
        "shards": {},
        "status": "planned",
    }
    manifest["query_hashes_by_year"] = {str(year): query_hash(filing_date_sql(schema_roles, year)) for year in years}
    if not execute:
        manifest["status"] = "dry_run_ok"
        atomic_write_json(paths.manifests / "filing_dates_extract.json", manifest)
        atomic_write_text(paths.reports / "filing_dates_extract_report.md", report_text(manifest))
        return manifest

    db = connect_wrds(statement_timeout="240s")
    try:
        for year in years:
            year_dir = data_root / f"year={year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            out_path = year_dir / "compustat_filing_dates.parquet"
            if out_path.exists():
                manifest["shards"][str(year)] = {"path": str(out_path), "skipped_existing": True, **parquet_info(out_path)}
                continue
            sql = filing_date_sql(schema_roles, year)
            started = now_iso()
            df = db.raw_sql(sql, date_cols=None)
            prepared = prepare_filing_dates(df)
            atomic_write_parquet(prepared, out_path)
            manifest["queries"].append(
                {
                    "name": f"compustat_filing_dates_{year}",
                    "hash": query_hash(sql),
                    "started_utc": started,
                    "finished_utc": now_iso(),
                    "rows": int(len(df)),
                    "prepared_rows": int(len(prepared)),
                    "columns": list(map(str, prepared.columns)),
                }
            )
            manifest["shards"][str(year)] = {"path": str(out_path), **parquet_info(out_path)}
            atomic_write_json(paths.manifests / "filing_dates_extract.json", manifest)
    finally:
        try:
            db.close()
        except Exception:
            pass

    manifest["status"] = "ok"
    atomic_write_json(paths.manifests / "filing_dates_extract.json", manifest)
    atomic_write_text(paths.reports / "filing_dates_extract_report.md", report_text(manifest))
    return manifest


def report_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Filing Dates Extract Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Raw run ID: `{manifest['raw_run_id']}`",
        f"- Status: `{manifest['status']}`",
        f"- Execute: `{manifest['execute']}`",
        "",
        "| Year | Rows | Path |",
        "|---|---:|---|",
    ]
    for year, info in manifest.get("shards", {}).items():
        lines.append(f"| {year} | {info.get('rows', '')} | `{info.get('path', '')}` |")
    if not manifest.get("shards"):
        lines.append("| planned only |  | dry run |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--raw-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--years", default=None)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    years = years_from_args(args.years, args.start_year, args.end_year)
    manifest = run_extract(project_root, args.raw_run_id, args.run_id, years, args.execute)
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'filing_dates_extract.json'}")
    return 0 if manifest["status"] in {"ok", "dry_run_ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
