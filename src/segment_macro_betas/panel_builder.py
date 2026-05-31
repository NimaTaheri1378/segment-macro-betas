from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_macro_betas.features.exposures import hhi, is_domestic_label
from segment_macro_betas.io_utils import atomic_write_json, atomic_write_parquet, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_shards(raw_root: Path, dataset: str) -> pd.DataFrame:
    paths = sorted(raw_root.glob(f"year=*/{dataset}.parquet"))
    if not paths:
        raise FileNotFoundError(f"No shards found for {dataset} under {raw_root}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def read_optional_shards(raw_root: Path, dataset: str) -> pd.DataFrame | None:
    paths = sorted(raw_root.glob(f"year=*/{dataset}.parquet"))
    if not paths:
        return None
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def clean_segments(segments: pd.DataFrame) -> pd.DataFrame:
    df = segments.copy()
    for col in ("srcdate", "datadate"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["gvkey"] = df["gvkey"].astype(str)
    df["geo_label"] = df["gareag"].where(df["gareag"].notna() & (df["gareag"].astype(str).str.len() > 0), df["gareat"])
    sales = pd.to_numeric(df["sales"], errors="coerce")
    revts = pd.to_numeric(df["revts"], errors="coerce")
    ias = pd.to_numeric(df["ias"], errors="coerce")
    df["segment_sales"] = sales.where(sales.notna(), revts).where(lambda s: s.notna(), ias)
    df = df[df["srcdate"].notna() & df["segment_sales"].notna() & (df["segment_sales"] > 0)].copy()
    grouped = (
        df.groupby(["gvkey", "srcdate", "geo_label"], dropna=False, as_index=False)
        .agg(segment_sales=("segment_sales", "sum"), segment_rows=("sid", "count"), segment_datadate=("datadate", "max"))
    )
    grouped["segment_sales_sum"] = grouped.groupby(["gvkey", "srcdate"])["segment_sales"].transform("sum")
    grouped["revenue_share"] = grouped["segment_sales"] / grouped["segment_sales_sum"]
    grouped["is_domestic"] = grouped["geo_label"].map(is_domestic_label)
    rows: list[dict[str, Any]] = []
    for (gvkey, srcdate), g in grouped.groupby(["gvkey", "srcdate"], sort=False):
        shares = g["revenue_share"].fillna(0.0).clip(lower=0)
        rows.append(
            {
                "gvkey": gvkey,
                "segment_srcdate": srcdate,
                "segment_datadate": g["segment_datadate"].max(),
                "segment_sales_sum": float(g["segment_sales_sum"].max()),
                "geo_count": int(g["geo_label"].nunique(dropna=True)),
                "foreign_share": float(g.loc[~g["is_domestic"], "revenue_share"].sum()),
                "domestic_share": float(g.loc[g["is_domestic"], "revenue_share"].sum()),
                "geo_hhi": hhi(shares),
                "top_geo_share": float(shares.max()) if len(shares) else np.nan,
                "segment_rows": int(g["segment_rows"].sum()),
            }
        )
    snapshots = pd.DataFrame(rows)
    return snapshots.sort_values(["gvkey", "segment_srcdate"]).drop_duplicates(["gvkey", "segment_srcdate"], keep="last")


def prepare_filing_dates(filing_dates: pd.DataFrame | None) -> pd.DataFrame | None:
    if filing_dates is None or filing_dates.empty:
        return None
    df = filing_dates.copy()
    df["gvkey"] = df["gvkey"].astype(str)
    for col in ("datadate", "fdate", "pdate", "filing_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df[df["datadate"].notna()].copy()
    if "filing_date" not in df.columns:
        fdate = df["fdate"] if "fdate" in df.columns else pd.Series(pd.NaT, index=df.index)
        pdate = df["pdate"] if "pdate" in df.columns else pd.Series(pd.NaT, index=df.index)
        df["filing_date"] = pdate.where(pdate.notna(), fdate)
    if "filing_date_source" not in df.columns:
        df["filing_date_source"] = "missing"
        if "pdate" in df.columns:
            df.loc[df["pdate"].notna(), "filing_date_source"] = "pdate"
        if "fdate" in df.columns:
            df.loc[(df["filing_date_source"] == "missing") & df["fdate"].notna(), "filing_date_source"] = "fdate"
    return df.sort_values(["gvkey", "datadate", "filing_date"]).drop_duplicates(["gvkey", "datadate"], keep="first")


def apply_segment_activation_dates(
    snapshots: pd.DataFrame,
    filing_dates: pd.DataFrame | None,
    *,
    activation_lag_days: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = snapshots.copy()
    out["segment_datadate"] = pd.to_datetime(out["segment_datadate"], errors="coerce")
    out["segment_srcdate"] = pd.to_datetime(out["segment_srcdate"], errors="coerce")
    filings = prepare_filing_dates(filing_dates)
    if filings is not None and not filings.empty:
        out = out.merge(
            filings[["gvkey", "datadate", "filing_date", "filing_date_source"]],
            left_on=["gvkey", "segment_datadate"],
            right_on=["gvkey", "datadate"],
            how="left",
        ).drop(columns=["datadate"], errors="ignore")
    else:
        out["filing_date"] = pd.NaT
        out["filing_date_source"] = "missing"
    out["segment_activation_base_date"] = out["filing_date"].where(out["filing_date"].notna(), out["segment_srcdate"])
    out["segment_activation_source"] = out["filing_date_source"].where(out["filing_date"].notna(), "srcdate_fallback")
    out["segment_activation_date"] = out["segment_activation_base_date"] + pd.to_timedelta(activation_lag_days, unit="D")
    checks = {
        "segment_snapshots": int(len(out)),
        "filing_date_matched_snapshots": int(out["filing_date"].notna().sum()),
        "filing_date_match_rate": float(out["filing_date"].notna().mean()) if len(out) else None,
        "activation_lag_days": int(activation_lag_days),
        "activation_source_counts": out["segment_activation_source"].value_counts(dropna=False).to_dict(),
    }
    return out, checks


def prepare_crsp(crsp: pd.DataFrame) -> pd.DataFrame:
    df = crsp.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
    df = df[df["date"].notna() & df["permno"].notna()].copy()
    df["permno"] = df["permno"].astype(int)
    for col in ("ret", "retx", "prc", "mktcap", "vol"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"], keep="last")
    return df


def prepare_ccm(ccm: pd.DataFrame) -> pd.DataFrame:
    df = ccm.copy()
    df["gvkey"] = df["gvkey"].astype(str)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
    df = df[df["permno"].notna()].copy()
    df["permno"] = df["permno"].astype(int)
    df["linkdt"] = pd.to_datetime(df["linkdt"], errors="coerce")
    df["linkenddt"] = pd.to_datetime(df["linkenddt"], errors="coerce").fillna(pd.Timestamp("2100-01-01"))
    df = df[df["linkdt"].notna()].copy()
    return df.drop_duplicates(["gvkey", "permno", "linkdt", "linkenddt", "linktype", "linkprim"])


def link_crsp_to_gvkey(crsp: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    linked = crsp.merge(ccm, on="permno", how="inner")
    linked = linked[(linked["date"] >= linked["linkdt"]) & (linked["date"] <= linked["linkenddt"])].copy()
    linked = linked.sort_values(["permno", "date", "linkprim", "linktype", "linkdt"])
    return linked.drop_duplicates(["permno", "date"], keep="last")


def merge_asof_by_key(left: pd.DataFrame, right: pd.DataFrame, *, by: str, left_on: str, right_on: str) -> pd.DataFrame:
    pieces = []
    right_groups = {key: group.sort_values(right_on) for key, group in right.groupby(by, sort=False)}
    for key, left_group in left.groupby(by, sort=False):
        rg = right_groups.get(key)
        if rg is None or rg.empty:
            chunk = left_group.copy()
            for col in right.columns:
                if col not in chunk.columns and col != by:
                    chunk[col] = pd.NA
            pieces.append(chunk)
            continue
        pieces.append(
            pd.merge_asof(
                left_group.sort_values(left_on),
                rg,
                by=by,
                left_on=left_on,
                right_on=right_on,
                direction="backward",
                allow_exact_matches=False,
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else left.iloc[0:0].copy()


def prepare_funda(funda: pd.DataFrame) -> pd.DataFrame:
    df = funda.copy()
    df["gvkey"] = df["gvkey"].astype(str)
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    for col in ("fdate", "pdate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df[df["datadate"].notna()].copy()
    fallback_date = df["datadate"] + pd.DateOffset(months=6)
    if "pdate" in df.columns or "fdate" in df.columns:
        pdate = df["pdate"] if "pdate" in df.columns else pd.Series(pd.NaT, index=df.index)
        fdate = df["fdate"] if "fdate" in df.columns else pd.Series(pd.NaT, index=df.index)
        df["funda_avail_date"] = pdate.where(pdate.notna(), fdate).where(lambda s: s.notna(), fallback_date)
        df["funda_avail_source"] = "fallback_6m"
        df.loc[pdate.notna(), "funda_avail_source"] = "pdate"
        df.loc[pdate.isna() & fdate.notna(), "funda_avail_source"] = "fdate"
    else:
        df["funda_avail_date"] = fallback_date
        df["funda_avail_source"] = "fallback_6m"
    for col in ("at", "sale", "revt", "ceq", "ni", "capx", "xrd", "dltt", "dlc", "prcc_f", "csho"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["gvkey", "funda_avail_date"]).drop_duplicates(["gvkey", "funda_avail_date"], keep="last")


def prepare_factors(factors: pd.DataFrame) -> pd.DataFrame:
    df = factors.copy()
    if "dateff" in df.columns:
        df["factor_date"] = pd.to_datetime(df["dateff"], errors="coerce")
    else:
        df["factor_date"] = pd.to_datetime(df["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    for col in ("mktrf", "smb", "hml", "rf", "umd"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].abs().max(skipna=True) and df[col].abs().max(skipna=True) > 1.0:
                df[col] = df[col] / 100.0
    return df.drop(columns=["date"], errors="ignore").drop_duplicates(["factor_date"], keep="last")


def build_panel(project_root: Path, raw_run_id: str, run_id: str) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    raw_root = ensure_within(project_root, project_root / "data" / "raw" / raw_run_id)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "raw_run_id": raw_run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "inputs": {},
        "checks": {},
        "outputs": {},
        "status": "started",
    }

    segments = read_shards(raw_root, "segments")
    crsp = read_shards(raw_root, "crsp_monthly")
    ccm = read_shards(raw_root, "ccm_links")
    funda = read_shards(raw_root, "compustat_funda")
    factors = read_shards(raw_root, "factors_monthly")
    filing_dates = read_optional_shards(raw_root, "compustat_filing_dates")
    manifest["inputs"] = {
        "segments_rows": int(len(segments)),
        "crsp_rows": int(len(crsp)),
        "ccm_rows": int(len(ccm)),
        "funda_rows": int(len(funda)),
        "factors_rows": int(len(factors)),
        "filing_date_rows": int(len(filing_dates)) if filing_dates is not None else 0,
    }

    snapshots_raw = clean_segments(segments)
    snapshots, activation_checks = apply_segment_activation_dates(snapshots_raw, filing_dates)
    crsp_prepped = prepare_crsp(crsp)
    ccm_prepped = prepare_ccm(ccm)
    linked = link_crsp_to_gvkey(crsp_prepped, ccm_prepped)
    panel = merge_asof_by_key(linked, snapshots, by="gvkey", left_on="date", right_on="segment_activation_date")
    panel = panel[panel["segment_activation_date"].notna()].copy()

    funda_prepped = prepare_funda(funda)
    panel = merge_asof_by_key(panel, funda_prepped, by="gvkey", left_on="date", right_on="funda_avail_date")
    factors_prepped = prepare_factors(factors)
    panel = panel.merge(factors_prepped, left_on="date", right_on="factor_date", how="left")
    if "rf" in panel.columns:
        panel["excess_ret"] = panel["ret"] - panel["rf"]
    panel = panel.sort_values(["permno", "date"])
    panel["next_month_ret"] = panel.groupby("permno")["ret"].shift(-1)
    if "rf" in panel.columns:
        panel["next_month_excess_ret"] = panel.groupby("permno")["excess_ret"].shift(-1)

    keep_cols = [
        "gvkey",
        "permno",
        "date",
        "ret",
        "retx",
        "excess_ret",
        "next_month_ret",
        "next_month_excess_ret",
        "prc",
        "mktcap",
        "vol",
        "segment_srcdate",
        "segment_datadate",
        "segment_activation_base_date",
        "segment_activation_date",
        "segment_activation_source",
        "filing_date",
        "filing_date_source",
        "foreign_share",
        "domestic_share",
        "geo_hhi",
        "geo_count",
        "top_geo_share",
        "segment_sales_sum",
        "at",
        "sale",
        "revt",
        "ceq",
        "ni",
        "capx",
        "xrd",
        "dltt",
        "dlc",
        "funda_avail_date",
        "funda_avail_source",
        "mktrf",
        "smb",
        "hml",
        "rf",
        "umd",
    ]
    keep_cols = [col for col in keep_cols if col in panel.columns]
    panel = panel[keep_cols]

    out_path = ensure_within(project_root, project_root / "data" / "interim" / run_id / "monthly_panel.parquet")
    summary_path = ensure_within(project_root, project_root / "artifacts" / "tables" / run_id / "monthly_panel_summary.csv")
    atomic_write_parquet(panel, out_path)
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(panel)},
            {"metric": "unique_gvkeys", "value": panel["gvkey"].nunique()},
            {"metric": "unique_permnos", "value": panel["permno"].nunique()},
            {"metric": "min_date", "value": str(panel["date"].min().date()) if len(panel) else None},
            {"metric": "max_date", "value": str(panel["date"].max().date()) if len(panel) else None},
            {"metric": "mean_foreign_share", "value": panel["foreign_share"].mean() if len(panel) else None},
            {"metric": "mean_geo_hhi", "value": panel["geo_hhi"].mean() if len(panel) else None},
            {"metric": "next_month_ret_nonmissing", "value": int(panel["next_month_ret"].notna().sum()) if "next_month_ret" in panel else 0},
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_summary = summary_path.with_suffix(".csv.tmp")
    summary.to_csv(tmp_summary, index=False)
    tmp_summary.replace(summary_path)

    activation_violations = int((panel["date"] <= panel["segment_activation_date"]).sum()) if len(panel) else 0
    manifest["checks"] = {
        "snapshot_rows": int(len(snapshots)),
        **activation_checks,
        "linked_crsp_rows": int(len(linked)),
        "panel_rows": int(len(panel)),
        "unique_gvkeys": int(panel["gvkey"].nunique()) if len(panel) else 0,
        "unique_permnos": int(panel["permno"].nunique()) if len(panel) else 0,
        "min_date": str(panel["date"].min().date()) if len(panel) else None,
        "max_date": str(panel["date"].max().date()) if len(panel) else None,
        "activation_rule_violations": activation_violations,
        "foreign_share_missing": int(panel["foreign_share"].isna().sum()) if len(panel) else 0,
        "factors_missing_rf": int(panel["rf"].isna().sum()) if "rf" in panel else None,
    }
    manifest["outputs"] = {"monthly_panel": str(out_path), "summary": str(summary_path)}
    manifest["status"] = "ok" if len(panel) > 0 and activation_violations == 0 else "needs_review"
    atomic_write_json(paths.manifests / "monthly_panel.json", manifest)
    atomic_write_text(paths.reports / "monthly_panel_report.md", report_text(manifest))
    return manifest


def report_text(manifest: dict[str, Any]) -> str:
    c = manifest["checks"]
    return "\n".join(
        [
            "# Monthly Panel Report",
            "",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Raw run ID: `{manifest['raw_run_id']}`",
            f"- Status: `{manifest['status']}`",
            f"- Panel rows: `{c.get('panel_rows')}`",
            f"- Unique GVKEYs: `{c.get('unique_gvkeys')}`",
            f"- Unique PERMNOs: `{c.get('unique_permnos')}`",
            f"- Date range: `{c.get('min_date')}` to `{c.get('max_date')}`",
            f"- Activation rule violations: `{c.get('activation_rule_violations')}`",
            f"- Missing RF rows: `{c.get('factors_missing_rf')}`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--raw-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = build_panel(project_root, args.raw_run_id, args.run_id)
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'monthly_panel.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
