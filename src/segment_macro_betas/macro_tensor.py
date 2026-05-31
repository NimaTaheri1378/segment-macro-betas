from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_macro_betas.io_utils import atomic_write_json, atomic_write_parquet, atomic_write_text
from segment_macro_betas.macro_engine import DEFAULT_FRED_SERIES
from segment_macro_betas.panel_builder import read_shards
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


DEFAULT_SERIES_AREA = {
    "FEDFUNDS": "USA",
    "CPIAUCSL": "USA",
    "UNRATE": "USA",
    "INDPRO": "USA",
    "DTWEXBGS": "USD",
}
AREA_RULES = [
    ("GLOBAL", ("GLOBAL", "WORLD", "ALL", "ALL_AREAS")),
    ("USD", ("USD", "DOLLAR", "TRADE WEIGHTED")),
    ("USA", ("USA", "UNITED STATES", "U.S.", "US", "N_AMER", "NORTH AMERICA")),
    ("CANADA", ("CAN", "CANADA")),
    ("CHINA", ("CHINA", "CHIN", "PRC")),
    ("EUROPE", ("EUROPE", "EURO", "EU", "EUR", "GERMANY", "FRANCE", "ITALY", "SPAIN", "UK", "UNITED KINGDOM")),
    ("ASIA_EX_CHINA", ("ASIA", "JAPAN", "KOREA", "TAIWAN", "HKG", "HONG KONG", "INDIA", "APAC")),
    ("LATAM", ("LATIN", "LATAM", "MEXICO", "BRAZIL", "ARGENTINA", "CHILE")),
    ("MIDDLE_EAST", ("MID_EAST", "MIDDLE EAST", "SAUDI", "UAE", "ISRAEL")),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slug(value: str) -> str:
    out = []
    previous_underscore = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            previous_underscore = False
        elif not previous_underscore:
            out.append("_")
            previous_underscore = True
    return "".join(out).strip("_") or "unknown"


def canonical_macro_area(label: Any) -> str:
    if pd.isna(label):
        return "UNKNOWN"
    text = str(label).strip().upper()
    if not text:
        return "UNKNOWN"
    for area, terms in AREA_RULES:
        if any(term in text for term in terms):
            return area
    return "OTHER_FOREIGN"


def series_name(series_id: Any) -> str:
    raw = str(series_id).strip().upper()
    return DEFAULT_FRED_SERIES.get(raw, slug(raw))


def prepare_segment_geo_tokens(segments: pd.DataFrame) -> pd.DataFrame:
    df = segments.copy()
    df["gvkey"] = df["gvkey"].astype(str)
    for col in ("srcdate", "datadate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    geo = df["gareag"].where(df["gareag"].notna() & (df["gareag"].astype(str).str.len() > 0), df["gareat"])
    df["geo_label"] = geo.fillna("UNKNOWN").astype(str)
    sales = pd.to_numeric(df.get("sales"), errors="coerce")
    revts = pd.to_numeric(df.get("revts"), errors="coerce")
    ias = pd.to_numeric(df.get("ias"), errors="coerce")
    df["segment_sales"] = sales.where(sales.notna(), revts).where(lambda s: s.notna(), ias)
    df = df[df["srcdate"].notna() & df["segment_sales"].notna() & (df["segment_sales"] > 0)].copy()
    grouped = (
        df.groupby(["gvkey", "srcdate", "geo_label"], as_index=False, dropna=False)
        .agg(segment_sales=("segment_sales", "sum"), segment_rows=("sid", "count"))
        .sort_values(["gvkey", "srcdate", "segment_sales"], ascending=[True, True, False])
    )
    grouped["snapshot_sales"] = grouped.groupby(["gvkey", "srcdate"])["segment_sales"].transform("sum")
    grouped["revenue_share"] = grouped["segment_sales"] / grouped["snapshot_sales"]
    grouped["macro_area"] = grouped["geo_label"].map(canonical_macro_area)
    grouped = grouped[grouped["revenue_share"].notna() & (grouped["revenue_share"] > 0)].copy()
    return grouped.rename(columns={"srcdate": "segment_srcdate"})


def infer_available_date(df: pd.DataFrame, release_lag_days: int) -> tuple[pd.Series, str, bool]:
    for col in ("available_date", "realtime_start", "release_date"):
        if col in df.columns:
            available = pd.to_datetime(df[col], errors="coerce")
            if available.notna().any():
                return available, col, True
    fallback = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(release_lag_days, unit="D")
    return fallback, "observation_date_fallback", False


def prepare_macro_states(macro: pd.DataFrame, *, release_lag_days: int = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    if macro.empty:
        cols = ["macro_area", "series_id", "series_name", "observation_date", "available_date", "value"]
        return pd.DataFrame(columns=cols), {"rows": 0, "vintage_safe": False, "availability_source": "empty"}

    df = macro.copy()
    if "date" not in df.columns:
        raise KeyError("Macro data must include a date column.")
    if "series_id" not in df.columns:
        raise KeyError("Macro data must include a series_id column.")
    if "value" not in df.columns:
        raise KeyError("Macro data must include a value column.")

    df["series_id"] = df["series_id"].astype(str).str.upper()
    df["series_name"] = df["series_id"].map(series_name)
    df["observation_date"] = pd.to_datetime(df["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "macro_area" in df.columns:
        df["macro_area"] = df["macro_area"].fillna(df["series_id"].map(DEFAULT_SERIES_AREA)).map(canonical_macro_area)
    elif "geo_label" in df.columns:
        df["macro_area"] = df["geo_label"].map(canonical_macro_area)
    else:
        df["macro_area"] = df["series_id"].map(DEFAULT_SERIES_AREA).fillna("GLOBAL")

    available, source, vintage_safe = infer_available_date(df, release_lag_days)
    df["available_date"] = available
    df = df[df["observation_date"].notna() & df["available_date"].notna() & df["value"].notna()].copy()
    df = df.sort_values(["macro_area", "series_name", "available_date", "observation_date"])
    df = df.drop_duplicates(["macro_area", "series_name", "available_date"], keep="last")
    df["value_delta_1m"] = df.groupby(["macro_area", "series_name"])["value"].diff(1)
    df["value_delta_12m"] = df.groupby(["macro_area", "series_name"])["value"].diff(12)
    checks = {
        "rows": int(len(df)),
        "series": sorted(df["series_name"].dropna().unique().tolist()),
        "macro_areas": sorted(df["macro_area"].dropna().unique().tolist()),
        "availability_source": source,
        "vintage_safe": bool(vintage_safe),
        "release_lag_days": int(release_lag_days),
        "lookahead_safe": bool(df["lookahead_safe"].all()) if "lookahead_safe" in df.columns and len(df) else bool(vintage_safe),
        "revision_safe": bool(df["revision_safe"].all()) if "revision_safe" in df.columns and len(df) else None,
        "timing_sources": sorted(df["timing_source"].dropna().astype(str).unique().tolist()) if "timing_source" in df.columns else [source],
    }
    return df, checks


def attach_macro_to_tokens(tokens: pd.DataFrame, macro_states: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = tokens.copy().reset_index(drop=True)
    out["_token_row_id"] = np.arange(len(out))
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    feature_cols: list[str] = []
    if macro_states.empty or out.empty:
        return out.drop(columns=["_token_row_id"]), feature_cols

    for name, states in macro_states.groupby("series_name", sort=True):
        value_col = f"macro_{slug(name)}"
        delta_1m_col = f"{value_col}_delta_1m"
        delta_12m_col = f"{value_col}_delta_12m"
        feature_cols.extend([value_col, delta_1m_col, delta_12m_col])
        joined_frames = []
        state_groups = {area: group.sort_values("available_date") for area, group in states.groupby("macro_area", sort=False)}
        global_state_group = state_groups.get("GLOBAL")
        for area, token_group in out.groupby("macro_area", sort=False):
            state_group = state_groups.get(area)
            if (state_group is None or state_group.empty) and global_state_group is not None:
                state_group = global_state_group
            if state_group is None or state_group.empty:
                chunk = token_group[["_token_row_id"]].copy()
                chunk[value_col] = np.nan
                chunk[delta_1m_col] = np.nan
                chunk[delta_12m_col] = np.nan
            else:
                merged = pd.merge_asof(
                    token_group.sort_values("date")[["_token_row_id", "date"]],
                    state_group[["available_date", "value", "value_delta_1m", "value_delta_12m"]].sort_values("available_date"),
                    left_on="date",
                    right_on="available_date",
                    direction="backward",
                    allow_exact_matches=True,
                )
                chunk = merged[["_token_row_id", "value", "value_delta_1m", "value_delta_12m"]].rename(
                    columns={"value": value_col, "value_delta_1m": delta_1m_col, "value_delta_12m": delta_12m_col}
                )
            joined_frames.append(chunk)
        joined = pd.concat(joined_frames, ignore_index=True).set_index("_token_row_id")
        out = out.join(joined, on="_token_row_id")
    return out.drop(columns=["_token_row_id"]), feature_cols


def aggregate_macro_features(token_macro: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["gvkey", "permno", "date"]
    if token_macro.empty:
        return pd.DataFrame(columns=keys), {"token_rows": 0, "firm_month_rows": 0, "macro_coverage_rate": None}
    weighted = token_macro[keys + ["revenue_share", "macro_area", "geo_label"]].copy()
    for col in feature_cols:
        weighted[f"segw_{col}"] = token_macro["revenue_share"] * pd.to_numeric(token_macro[col], errors="coerce")

    agg_map: dict[str, tuple[str, str]] = {
        "macro_token_rows": ("revenue_share", "size"),
        "macro_geo_count": ("geo_label", "nunique"),
        "macro_area_count": ("macro_area", "nunique"),
    }
    for col in feature_cols:
        agg_map[f"segment_{col}"] = (f"segw_{col}", "sum")
    out = weighted.groupby(keys, as_index=False).agg(**agg_map)

    has_macro = token_macro[feature_cols].notna().any(axis=1) if feature_cols else pd.Series(False, index=token_macro.index)
    coverage = (
        token_macro.assign(_covered_share=token_macro["revenue_share"].where(has_macro, 0.0))
        .groupby(keys, as_index=False)
        .agg(macro_revenue_share_covered=("_covered_share", "sum"))
    )
    out = out.merge(coverage, on=keys, how="left")
    checks = {
        "token_rows": int(len(token_macro)),
        "firm_month_rows": int(len(out)),
        "macro_feature_count": int(len(feature_cols)),
        "macro_coverage_rate": float((out["macro_revenue_share_covered"] > 0).mean()) if len(out) else None,
        "mean_macro_revenue_share_covered": float(out["macro_revenue_share_covered"].mean()) if len(out) else None,
    }
    return out, checks


def build_macro_tensor(
    panel: pd.DataFrame,
    segments: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    release_lag_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = panel.copy()
    frame["gvkey"] = frame["gvkey"].astype(str)
    frame["permno"] = pd.to_numeric(frame["permno"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["segment_srcdate"] = pd.to_datetime(frame["segment_srcdate"], errors="coerce")
    frame = frame[frame["gvkey"].notna() & frame["permno"].notna() & frame["date"].notna() & frame["segment_srcdate"].notna()].copy()
    frame["permno"] = frame["permno"].astype(int)

    segment_tokens = prepare_segment_geo_tokens(segments)
    token_frame = frame[["gvkey", "permno", "date", "segment_srcdate"]].merge(
        segment_tokens[["gvkey", "segment_srcdate", "geo_label", "macro_area", "revenue_share", "segment_sales", "segment_rows"]],
        on=["gvkey", "segment_srcdate"],
        how="inner",
    )
    panel_keys_with_tokens = token_frame[["gvkey", "permno", "date"]].drop_duplicates().shape[0] if len(token_frame) else 0
    macro_states, macro_checks = prepare_macro_states(macro, release_lag_days=release_lag_days)
    token_macro, feature_cols = attach_macro_to_tokens(token_frame, macro_states)
    aggregated, agg_checks = aggregate_macro_features(token_macro, feature_cols)
    tensor_panel = frame.merge(aggregated, on=["gvkey", "permno", "date"], how="left")
    checks = {
        "panel_rows": int(len(frame)),
        "segment_token_rows": int(len(segment_tokens)),
        "joined_token_rows": int(len(token_frame)),
        "panel_rows_with_tokens": int(panel_keys_with_tokens),
        "joined_token_match_rate": float(panel_keys_with_tokens / len(frame)) if len(frame) else None,
        "joined_tokens_per_panel_row": float(len(token_frame) / len(frame)) if len(frame) else None,
        "macro": macro_checks,
        "aggregation": agg_checks,
    }
    return tensor_panel, token_macro, checks


def report_text(manifest: dict[str, Any]) -> str:
    checks = manifest.get("checks", {})
    macro = checks.get("macro", {})
    agg = checks.get("aggregation", {})
    return "\n".join(
        [
            "# Macro Tensor Report",
            "",
            f"- Run ID: `{manifest['run_id']}`",
            f"- Status: `{manifest['status']}`",
            f"- Panel run: `{manifest['inputs']['panel_run_id']}`",
            f"- Raw segment run: `{manifest['inputs']['raw_run_id']}`",
            f"- Macro run: `{manifest['inputs']['macro_run_id']}`",
            f"- Panel rows: `{checks.get('panel_rows')}`",
            f"- Token rows: `{checks.get('joined_token_rows')}`",
            f"- Firm-month rows with macro features: `{agg.get('firm_month_rows')}`",
            f"- Macro features: `{agg.get('macro_feature_count')}`",
            f"- Macro vintage-safe flag: `{macro.get('vintage_safe')}`",
            f"- Macro availability source: `{macro.get('availability_source')}`",
            "",
            "This report stores counts and timing flags only; it does not store API credentials.",
            "",
        ]
    )


def run_macro_tensor(project_root: Path, run_id: str, raw_run_id: str, panel_run_id: str, macro_run_id: str, macro_dataset: str, release_lag_days: int) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    raw_root = ensure_within(project_root, project_root / "data" / "raw" / raw_run_id)
    panel_path = ensure_within(project_root, project_root / "data" / "interim" / panel_run_id / "monthly_panel.parquet")
    macro_path = ensure_within(project_root, project_root / "data" / "raw" / macro_run_id / f"{macro_dataset}.parquet")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "inputs": {
            "raw_run_id": raw_run_id,
            "panel_run_id": panel_run_id,
            "macro_run_id": macro_run_id,
            "macro_dataset": macro_dataset,
            "release_lag_days": int(release_lag_days),
        },
        "outputs": {},
        "checks": {},
        "status": "started",
    }
    panel = pd.read_parquet(panel_path)
    segments = read_shards(raw_root, "segments")
    macro = pd.read_parquet(macro_path)
    tensor_panel, token_macro, checks = build_macro_tensor(panel, segments, macro, release_lag_days=release_lag_days)

    tensor_path = ensure_within(project_root, project_root / "data" / "interim" / run_id / "macro_tensor_panel.parquet")
    token_path = ensure_within(project_root, project_root / "data" / "interim" / run_id / "segment_macro_tokens.parquet")
    summary_path = ensure_within(project_root, project_root / "artifacts" / "tables" / run_id / "macro_tensor_summary.csv")
    atomic_write_parquet(tensor_panel, tensor_path)
    atomic_write_parquet(token_macro, token_path)
    summary = pd.DataFrame(
        [
            {"metric": "panel_rows", "value": checks["panel_rows"]},
            {"metric": "joined_token_rows", "value": checks["joined_token_rows"]},
            {"metric": "firm_month_rows", "value": checks["aggregation"]["firm_month_rows"]},
            {"metric": "macro_feature_count", "value": checks["aggregation"]["macro_feature_count"]},
            {"metric": "macro_coverage_rate", "value": checks["aggregation"]["macro_coverage_rate"]},
            {"metric": "macro_vintage_safe", "value": checks["macro"]["vintage_safe"]},
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    manifest["outputs"] = {"tensor_panel": str(tensor_path), "token_tensor": str(token_path), "summary": str(summary_path)}
    manifest["checks"] = checks
    manifest["status"] = "ok"
    atomic_write_json(paths.manifests / "macro_tensor.json", manifest)
    atomic_write_text(paths.reports / "macro_tensor_report.md", report_text(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-run-id", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--macro-run-id", required=True)
    parser.add_argument("--macro-dataset", default="macro_official_monthly")
    parser.add_argument("--release-lag-days", type=int, default=0)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_macro_tensor(
        project_root,
        args.run_id,
        args.raw_run_id,
        args.panel_run_id,
        args.macro_run_id,
        args.macro_dataset,
        args.release_lag_days,
    )
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'macro_tensor.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
