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


def read_sql(db: Any, sql: str, manifest: dict[str, Any], name: str) -> pd.DataFrame:
    started = now_iso()
    df = db.raw_sql(sql, date_cols=None)
    manifest["queries"].append(
        {
            "name": name,
            "hash": query_hash(sql),
            "started_utc": started,
            "finished_utc": now_iso(),
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
        }
    )
    return df


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


def sql_contracts(schema_roles: dict[str, dict[str, str]], year: int) -> dict[str, str]:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    next_q1 = f"{year + 1}-03-31"
    geo = schema_roles["historical_segments_geo"]
    vals = schema_roles["historical_segments_values"]
    crsp = schema_roles["crsp_monthly_stock"]
    daily = schema_roles["crsp_daily_stock"]
    ccm = schema_roles["ccm_link_history"]
    funda = schema_roles["compustat_fundamentals_annual"]
    factors = schema_roles["benchmark_factors_wrds"]

    return {
        "segments": f"""
select
    g.gvkey,
    g.stype,
    g.sid,
    g.gareag,
    g.gareat,
    g.srcdate,
    v.datadate,
    v.sales,
    v.revts,
    v.ias
from {table_ref(geo['library'], geo['table'])} as g
join {table_ref(vals['library'], vals['table'])} as v
  on g.gvkey = v.gvkey
 and g.stype = v.stype
 and g.sid = v.sid
 and g.srcdate = v.srcdate
where g.srcdate between {qliteral(start)} and {qliteral(end)}
  and coalesce(v.sales, v.revts, v.ias) is not null
""".strip(),
        "crsp_monthly": f"""
select
    permno,
    permco,
    siccd,
    yyyymm,
    mthcaldt as date,
    mthret as ret,
    mthretx as retx,
    mthprc as prc,
    mthcap as mktcap,
    mthvol as vol,
    cusip,
    ticker,
    sharetype,
    securitytype,
    securitysubtype,
    usincflg
from {table_ref(crsp['library'], crsp['table'])}
where mthcaldt between {qliteral(start)} and {qliteral(next_q1)}
""".strip(),
        "crsp_daily_events": f"""
select
    permno,
    dlycaldt as date,
    dlyret as ret,
    dlyretx as retx,
    dlyprc as prc,
    dlycap as mktcap,
    dlyvol as vol
from {table_ref(daily['library'], daily['table'])}
where dlycaldt between {qliteral(start)} and {qliteral(end)}
""".strip(),
        "compustat_funda": f"""
select
    gvkey,
    datadate,
    fyear,
    fyr,
    tic,
    conm,
    at,
    sale,
    revt,
    ceq,
    ni,
    capx,
    xrd,
    dltt,
    dlc,
    prcc_f,
    csho
from {table_ref(funda['library'], funda['table'])}
where datadate between {qliteral(str(year - 2) + '-01-01')} and {qliteral(end)}
  and indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
""".strip(),
        "ccm_links": f"""
select
    gvkey,
    lpermno as permno,
    lpermco as permco,
    linkdt,
    linkenddt,
    linktype,
    linkprim,
    usedflag
from {table_ref(ccm['library'], ccm['table'])}
where linkdt <= {qliteral(next_q1)}
  and (linkenddt is null or linkenddt >= {qliteral(start)})
  and linktype in ('LU','LC')
  and linkprim in ('P','C')
""".strip(),
        "factors_monthly": f"""
select *
from {table_ref(factors['library'], factors['table'])}
where date between {qliteral(start)} and {qliteral(next_q1)}
""".strip(),
    }


def summarize_frame(df: pd.DataFrame, date_col: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(df)), "columns": list(map(str, df.columns))}
    if date_col and date_col in df.columns and len(df):
        dates = pd.to_datetime(df[date_col], errors="coerce")
        summary["min_date"] = str(dates.min().date()) if dates.notna().any() else None
        summary["max_date"] = str(dates.max().date()) if dates.notna().any() else None
    return summary


def run_extract(project_root: Path, run_id: str, years: list[int], include_daily: bool, execute: bool) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    schema_roles = load_schema_roles(project_root / "configs" / "schema_map.yml")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "years": years,
        "include_daily": include_daily,
        "execute": execute,
        "pgpass": pgpass_metadata(),
        "queries": [],
        "shards": {},
        "status": "planned",
    }

    data_root = ensure_within(project_root, project_root / "data" / "raw" / run_id)
    contracts_by_year: dict[str, dict[str, str]] = {}
    for year in years:
        contracts = sql_contracts(schema_roles, year)
        if not include_daily:
            contracts.pop("crsp_daily_events")
        contracts_by_year[str(year)] = {name: query_hash(sql) for name, sql in contracts.items()}
    manifest["query_hashes_by_year"] = contracts_by_year

    if not execute:
        manifest["status"] = "dry_run_ok"
        atomic_write_json(paths.manifests / "full_extract.json", manifest)
        atomic_write_text(paths.reports / "full_extract_report.md", report_text(manifest))
        return manifest

    db = connect_wrds(statement_timeout="300s")
    try:
        for year in years:
            year_dir = data_root / f"year={year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            contracts = sql_contracts(schema_roles, year)
            if not include_daily:
                contracts.pop("crsp_daily_events")
            manifest["shards"][str(year)] = {}
            for name, sql in contracts.items():
                out_path = year_dir / f"{name}.parquet"
                if out_path.exists():
                    manifest["shards"][str(year)][name] = {"path": str(out_path), "skipped_existing": True, **parquet_info(out_path)}
                    continue
                df = read_sql(db, sql, manifest, f"{name}_{year}")
                atomic_write_parquet(df, out_path)
                date_col = "date" if "date" in df.columns else ("srcdate" if "srcdate" in df.columns else ("datadate" if "datadate" in df.columns else None))
                manifest["shards"][str(year)][name] = {"path": str(out_path), **summarize_frame(df, date_col)}
                atomic_write_json(paths.manifests / "full_extract.json", manifest)
    finally:
        try:
            db.close()
        except Exception:
            pass

    manifest["status"] = "ok"
    atomic_write_json(paths.manifests / "full_extract.json", manifest)
    atomic_write_text(paths.reports / "full_extract_report.md", report_text(manifest))
    return manifest


def report_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Full Extract Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Status: `{manifest['status']}`",
        f"- Years: `{manifest['years']}`",
        f"- Include daily: `{manifest['include_daily']}`",
        f"- Execute: `{manifest['execute']}`",
        "",
        "## Shards",
        "",
        "| Year | Dataset | Rows | Path |",
        "|---|---|---:|---|",
    ]
    for year, datasets in manifest.get("shards", {}).items():
        for name, info in datasets.items():
            lines.append(f"| {year} | `{name}` | {info.get('rows', '')} | `{info.get('path', '')}` |")
    if not manifest.get("shards"):
        lines.append("| planned only | query contracts |  | dry run |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--years", default=None, help="Comma-separated years or ranges, e.g. 2006,2008-2010")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--include-daily", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    years = years_from_args(args.years, args.start_year, args.end_year)
    manifest = run_extract(project_root, args.run_id, years, args.include_daily, args.execute)
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'full_extract.json'}")
    return 0 if manifest["status"] in {"ok", "dry_run_ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
