from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from segment_macro_betas.config import load_schema_roles
from segment_macro_betas.features.exposures import hhi, is_domestic_label
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, make_run_paths, require_project_root, resolve_project_root
from segment_macro_betas.wrds_access import connect_wrds, pgpass_metadata, qliteral, query_hash, table_ref


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def in_clause(values: list[Any]) -> str:
    if not values:
        return "('')"
    return "(" + ", ".join(qliteral(str(value)) for value in values) + ")"


def numeric_in_clause(values: list[int]) -> str:
    if not values:
        return "(-1)"
    return "(" + ", ".join(str(int(value)) for value in values) + ")"


def read_sql(db: Any, sql: str, manifest: dict[str, Any], name: str) -> pd.DataFrame:
    started = now_iso()
    df = db.raw_sql(sql, date_cols=None)
    manifest["queries"].append(
        {
            "name": name,
            "hash": query_hash(sql),
            "sql": sql,
            "started_utc": started,
            "finished_utc": now_iso(),
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
        }
    )
    return df


def build_segment_exposures(raw: pd.DataFrame, max_firms: int) -> pd.DataFrame:
    df = raw.copy()
    for col in ("srcdate", "datadate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["geo_label"] = df["gareag"].where(df["gareag"].notna() & (df["gareag"].astype(str).str.len() > 0), df["gareat"])
    sales = pd.to_numeric(df["sales"], errors="coerce")
    revts = pd.to_numeric(df["revts"], errors="coerce")
    ias = pd.to_numeric(df["ias"], errors="coerce")
    df["segment_sales"] = sales.where(sales.notna(), revts).where(lambda s: s.notna(), ias)
    df = df[df["segment_sales"].notna() & (df["segment_sales"] > 0) & df["srcdate"].notna()].copy()

    firm_sales = df.groupby("gvkey", as_index=False)["segment_sales"].sum().sort_values("segment_sales", ascending=False)
    keep_gvkeys = firm_sales.head(max_firms)["gvkey"].astype(str).tolist()
    df = df[df["gvkey"].astype(str).isin(keep_gvkeys)].copy()
    group_cols = ["gvkey", "srcdate"]
    df["segment_sales_sum"] = df.groupby(group_cols)["segment_sales"].transform("sum")
    df["revenue_share"] = df["segment_sales"] / df["segment_sales_sum"]
    df["is_domestic"] = df["geo_label"].map(is_domestic_label)
    return df


def build_firm_month_panel(exposures: pd.DataFrame, ccm: pd.DataFrame, crsp: pd.DataFrame, funda: pd.DataFrame) -> pd.DataFrame:
    ccm = ccm.copy()
    for col in ("linkdt", "linkenddt"):
        ccm[col] = pd.to_datetime(ccm[col], errors="coerce")
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2100-01-01"))
    ccm["permno"] = pd.to_numeric(ccm["permno"], errors="coerce").astype("Int64")

    linked = exposures.merge(ccm, on="gvkey", how="inner")
    linked = linked[(linked["srcdate"] >= linked["linkdt"]) & (linked["srcdate"] <= linked["linkenddt"])].copy()
    linked = linked[linked["permno"].notna()].copy()
    linked["permno"] = linked["permno"].astype(int)

    crsp = crsp.copy()
    crsp["date"] = pd.to_datetime(crsp["date"], errors="coerce")
    for col in ("ret", "retx", "prc", "mktcap", "vol"):
        if col in crsp.columns:
            crsp[col] = pd.to_numeric(crsp[col], errors="coerce")

    panel = linked.merge(crsp, on="permno", how="inner")
    panel = panel[(panel["date"] > panel["srcdate"]) & (panel["date"] <= panel["srcdate"] + pd.DateOffset(months=12))].copy()
    if panel.empty:
        return panel

    feature_rows = []
    for keys, group in panel.groupby(["gvkey", "permno", "srcdate", "date"], dropna=False):
        weights = group["revenue_share"].fillna(0.0)
        domestic_share = float(group.loc[group["is_domestic"], "revenue_share"].sum())
        foreign_share = float(group.loc[~group["is_domestic"], "revenue_share"].sum())
        first = group.iloc[0]
        feature_rows.append(
            {
                "gvkey": keys[0],
                "permno": int(keys[1]),
                "segment_srcdate": keys[2],
                "date": keys[3],
                "ret": first.get("ret"),
                "retx": first.get("retx"),
                "prc": first.get("prc"),
                "mktcap": first.get("mktcap"),
                "vol": first.get("vol"),
                "foreign_share": foreign_share,
                "domestic_share": domestic_share,
                "geo_hhi": hhi(weights),
                "geo_count": int(group["geo_label"].nunique(dropna=True)),
                "segment_sales_sum": float(group["segment_sales_sum"].max()),
            }
        )
    out = pd.DataFrame(feature_rows)

    if not funda.empty:
        f = funda.copy()
        f["datadate"] = pd.to_datetime(f["datadate"], errors="coerce")
        for col in ("at", "sale", "revt", "ceq", "ni", "capx", "xrd", "dltt", "dlc"):
            if col in f.columns:
                f[col] = pd.to_numeric(f[col], errors="coerce")
        f = f.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey"], keep="last")
        out = out.merge(f, on="gvkey", how="left", suffixes=("", "_funda"))

    out = out.sort_values(["permno", "date", "segment_srcdate"]).reset_index(drop=True)
    out["next_month_ret"] = out.groupby("permno")["ret"].shift(-1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-year", type=int, default=2019)
    parser.add_argument("--max-firms", type=int, default=40)
    parser.add_argument("--max-segment-rows", type=int, default=50_000)
    args = parser.parse_args()

    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    run_paths = make_run_paths(project_root, args.run_id)
    schema_roles = load_schema_roles(project_root / "configs" / "schema_map.yml")

    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "sample_year": args.sample_year,
        "max_firms": args.max_firms,
        "max_segment_rows": args.max_segment_rows,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pgpass": pgpass_metadata(),
        "queries": [],
        "outputs": {},
        "checks": {},
    }

    start = f"{args.sample_year}-01-01"
    end = f"{args.sample_year}-12-31"
    crsp_end = f"{args.sample_year + 1}-03-31"

    geo = schema_roles["historical_segments_geo"]
    vals = schema_roles["historical_segments_values"]
    ccm_role = schema_roles["ccm_link_history"]
    crsp_role = schema_roles["crsp_monthly_stock"]
    funda_role = schema_roles["compustat_fundamentals_annual"]

    segment_sql = f"""
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
  and g.gareat is not null
  and coalesce(v.sales, v.revts, v.ias) is not null
limit {int(args.max_segment_rows)}
""".strip()

    db = connect_wrds(statement_timeout="180s")
    try:
        raw_segments = read_sql(db, segment_sql, manifest, "segment_geo_values_smoke")
        exposures = build_segment_exposures(raw_segments, args.max_firms)
        gvkeys = sorted(exposures["gvkey"].astype(str).unique().tolist())
        manifest["checks"]["segment_rows_raw"] = int(len(raw_segments))
        manifest["checks"]["segment_rows_after_cleaning"] = int(len(exposures))
        manifest["checks"]["segment_unique_gvkeys"] = int(len(gvkeys))

        ccm_sql = f"""
select
    gvkey,
    lpermno as permno,
    linkdt,
    linkenddt,
    linktype,
    linkprim
from {table_ref(ccm_role['library'], ccm_role['table'])}
where gvkey in {in_clause(gvkeys)}
  and linkdt <= {qliteral(crsp_end)}
  and (linkenddt is null or linkenddt >= {qliteral(start)})
  and linktype in ('LU','LC')
  and linkprim in ('P','C')
""".strip()
        ccm = read_sql(db, ccm_sql, manifest, "ccm_link_smoke")
        permnos = sorted(pd.to_numeric(ccm["permno"], errors="coerce").dropna().astype(int).unique().tolist())
        manifest["checks"]["ccm_rows"] = int(len(ccm))
        manifest["checks"]["ccm_unique_permnos"] = int(len(permnos))

        crsp_sql = f"""
select
    permno,
    mthcaldt as date,
    mthret as ret,
    mthretx as retx,
    mthprc as prc,
    mthcap as mktcap,
    mthvol as vol,
    sharetype,
    securitytype,
    securitysubtype,
    usincflg
from {table_ref(crsp_role['library'], crsp_role['table'])}
where mthcaldt between {qliteral(start)} and {qliteral(crsp_end)}
  and permno in {numeric_in_clause(permnos)}
""".strip()
        crsp = read_sql(db, crsp_sql, manifest, "crsp_monthly_smoke")
        manifest["checks"]["crsp_rows"] = int(len(crsp))

        funda_sql = f"""
select
    gvkey,
    datadate,
    fyear,
    at,
    sale,
    revt,
    ceq,
    ni,
    capx,
    xrd,
    dltt,
    dlc
from {table_ref(funda_role['library'], funda_role['table'])}
where gvkey in {in_clause(gvkeys)}
  and datadate between {qliteral(str(args.sample_year - 2) + '-01-01')} and {qliteral(end)}
  and indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
""".strip()
        funda = read_sql(db, funda_sql, manifest, "compustat_funda_smoke")
    finally:
        try:
            db.close()
        except Exception:
            pass

    panel = build_firm_month_panel(exposures, ccm, crsp, funda)
    exposures_path = run_paths.data_interim / "segment_exposures.parquet"
    panel_path = run_paths.data_interim / "smoke_panel.parquet"
    summary_path = run_paths.artifacts_tables / "smoke_summary.csv"
    exposures.to_parquet(exposures_path, index=False)
    panel.to_parquet(panel_path, index=False)

    summary = pd.DataFrame(
        [
            {"metric": "segment_rows", "value": len(exposures)},
            {"metric": "firm_month_rows", "value": len(panel)},
            {"metric": "unique_gvkeys", "value": panel["gvkey"].nunique() if not panel.empty else 0},
            {"metric": "unique_permnos", "value": panel["permno"].nunique() if not panel.empty else 0},
            {"metric": "mean_foreign_share", "value": panel["foreign_share"].mean() if not panel.empty else None},
            {"metric": "mean_geo_hhi", "value": panel["geo_hhi"].mean() if not panel.empty else None},
        ]
    )
    summary.to_csv(summary_path, index=False)

    manifest["outputs"] = {
        "segment_exposures": str(exposures_path),
        "smoke_panel": str(panel_path),
        "summary": str(summary_path),
    }
    manifest["checks"].update(
        {
            "panel_rows": int(len(panel)),
            "panel_unique_gvkeys": int(panel["gvkey"].nunique()) if not panel.empty else 0,
            "panel_unique_permnos": int(panel["permno"].nunique()) if not panel.empty else 0,
            "panel_min_date": str(panel["date"].min().date()) if not panel.empty else None,
            "panel_max_date": str(panel["date"].max().date()) if not panel.empty else None,
            "activation_rule_violations": int((panel["date"] <= panel["segment_srcdate"]).sum()) if not panel.empty else 0,
        }
    )
    manifest["status"] = "ok" if manifest["checks"]["panel_rows"] > 0 and manifest["checks"]["activation_rule_violations"] == 0 else "needs_review"

    manifest_path = run_paths.manifests / "smoke_panel.json"
    report_path = run_paths.reports / "smoke_panel_report.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Smoke Panel Report",
                "",
                f"- Run ID: `{args.run_id}`",
                f"- Status: `{manifest['status']}`",
                f"- Sample year: `{args.sample_year}`",
                f"- Segment rows after cleaning: `{manifest['checks']['segment_rows_after_cleaning']}`",
                f"- Panel rows: `{manifest['checks']['panel_rows']}`",
                f"- Unique GVKEYs: `{manifest['checks']['panel_unique_gvkeys']}`",
                f"- Unique PERMNOs: `{manifest['checks']['panel_unique_permnos']}`",
                f"- Panel dates: `{manifest['checks']['panel_min_date']}` to `{manifest['checks']['panel_max_date']}`",
                f"- Activation rule violations: `{manifest['checks']['activation_rule_violations']}`",
                "",
                "This is a deliberately small WRDS smoke panel. It validates the schema contract,",
                "identifier bridge, point-in-time activation direction, and private artifact paths.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    print(f"panel={panel_path}")
    print(f"status={manifest['status']}")
    return 0 if manifest["status"] == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
