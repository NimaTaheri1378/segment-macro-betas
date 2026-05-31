from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_macro_betas.io_utils import atomic_write_json, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_quintiles(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    if valid.nunique() < 5 or len(valid) < 25:
        return out
    ranks = valid.rank(method="first")
    out.loc[valid.index] = pd.qcut(ranks, 5, labels=False).astype("int64") + 1
    return out


def t_stat(series: pd.Series) -> float | None:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 3:
        return None
    sd = x.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return None
    return float(x.mean() / (sd / np.sqrt(len(x))))


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
        }
    )
    return plt


def run_baselines(project_root: Path, panel_run_id: str, run_id: str) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    panel_path = ensure_within(project_root, project_root / "data" / "interim" / panel_run_id / "monthly_panel.parquet")
    tables_dir = ensure_within(project_root, project_root / "artifacts" / "tables" / run_id)
    figures_dir = ensure_within(project_root, project_root / "artifacts" / "figures_static" / run_id)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel[
        panel["date"].notna()
        & panel["foreign_share"].notna()
        & panel["next_month_excess_ret"].notna()
        & (panel["date"] <= pd.Timestamp("2025-12-31"))
    ].copy()
    panel["foreign_quintile"] = panel.groupby("date", group_keys=False)["foreign_share"].apply(safe_quintiles)
    sort_input = panel[panel["foreign_quintile"].notna()].copy()
    sort_input["foreign_quintile"] = sort_input["foreign_quintile"].astype(int)

    sort_returns = (
        sort_input.groupby(["date", "foreign_quintile"], as_index=False)
        .agg(
            ew_next_excess_ret=("next_month_excess_ret", "mean"),
            n=("permno", "nunique"),
            mean_foreign_share=("foreign_share", "mean"),
            mean_geo_hhi=("geo_hhi", "mean"),
        )
        .sort_values(["date", "foreign_quintile"])
    )
    wide = sort_returns.pivot(index="date", columns="foreign_quintile", values="ew_next_excess_ret")
    spread = (wide.get(5) - wide.get(1)).rename("q5_minus_q1").reset_index()
    spread["cum_q5_minus_q1"] = (1.0 + spread["q5_minus_q1"].fillna(0.0)).cumprod() - 1.0

    rank_ic_rows = []
    fmb_rows = []
    for date, group in panel.groupby("date", sort=True):
        g = group[["foreign_share", "geo_hhi", "next_month_excess_ret"]].dropna()
        if len(g) < 25 or g["foreign_share"].nunique() < 3:
            continue
        rank_ic_rows.append(
            {
                "date": date,
                "rank_ic_foreign_share": g["foreign_share"].rank().corr(g["next_month_excess_ret"].rank()),
                "rank_ic_geo_hhi": g["geo_hhi"].rank().corr(g["next_month_excess_ret"].rank()),
                "n": len(g),
            }
        )
        x = g["foreign_share"].to_numpy(dtype=float)
        y = g["next_month_excess_ret"].to_numpy(dtype=float)
        x = x - x.mean()
        denom = float(np.dot(x, x))
        beta = float(np.dot(x, y - y.mean()) / denom) if denom > 0 else np.nan
        fmb_rows.append({"date": date, "beta_foreign_share": beta, "n": len(g)})

    rank_ic = pd.DataFrame(rank_ic_rows)
    fmb = pd.DataFrame(fmb_rows)
    summary = pd.DataFrame(
        [
            {"metric": "panel_rows_used", "value": len(panel)},
            {"metric": "sort_rows_used", "value": len(sort_input)},
            {"metric": "months_with_sorts", "value": sort_returns["date"].nunique()},
            {"metric": "mean_q5_minus_q1", "value": spread["q5_minus_q1"].mean()},
            {"metric": "t_q5_minus_q1", "value": t_stat(spread["q5_minus_q1"])},
            {"metric": "mean_rank_ic_foreign_share", "value": rank_ic["rank_ic_foreign_share"].mean() if len(rank_ic) else None},
            {"metric": "t_rank_ic_foreign_share", "value": t_stat(rank_ic["rank_ic_foreign_share"]) if len(rank_ic) else None},
            {"metric": "mean_fmb_beta_foreign_share", "value": fmb["beta_foreign_share"].mean() if len(fmb) else None},
            {"metric": "t_fmb_beta_foreign_share", "value": t_stat(fmb["beta_foreign_share"]) if len(fmb) else None},
        ]
    )

    sort_returns_path = tables_dir / "foreign_share_quintile_returns.csv"
    spread_path = tables_dir / "foreign_share_spread.csv"
    rank_ic_path = tables_dir / "rank_ic.csv"
    fmb_path = tables_dir / "fmb_foreign_share.csv"
    summary_path = tables_dir / "baseline_summary.csv"
    sort_returns.to_csv(sort_returns_path, index=False)
    spread.to_csv(spread_path, index=False)
    rank_ic.to_csv(rank_ic_path, index=False)
    fmb.to_csv(fmb_path, index=False)
    summary.to_csv(summary_path, index=False)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(spread["date"], spread["cum_q5_minus_q1"], color="#2D8A7D", linewidth=2.0)
    ax.set_title("Foreign-Share Q5 - Q1 Cumulative Return")
    ax.set_ylabel("Cumulative return")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    spread_fig = figures_dir / "foreign_share_spread_cumulative.png"
    fig.savefig(spread_fig)
    plt.close(fig)

    if len(rank_ic):
        ric = rank_ic.copy()
        ric["rolling_12m"] = ric["rank_ic_foreign_share"].rolling(12, min_periods=3).mean()
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(ric["date"], ric["rank_ic_foreign_share"], color="#C27D38", alpha=0.45, width=20)
        ax.plot(ric["date"], ric["rolling_12m"], color="#384B70", linewidth=2.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Monthly Rank IC: Foreign Share")
        ax.set_ylabel("Rank IC")
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        ic_fig = figures_dir / "foreign_share_rank_ic.png"
        fig.savefig(ic_fig)
        plt.close(fig)
    else:
        ic_fig = None

    manifest = {
        "run_id": run_id,
        "panel_run_id": panel_run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "inputs": {"panel": str(panel_path), "panel_rows_after_filter": int(len(panel))},
        "checks": {
            "sort_rows": int(len(sort_input)),
            "months_with_sorts": int(sort_returns["date"].nunique()) if len(sort_returns) else 0,
            "rank_ic_months": int(len(rank_ic)),
            "fmb_months": int(len(fmb)),
        },
        "outputs": {
            "sort_returns": str(sort_returns_path),
            "spread": str(spread_path),
            "rank_ic": str(rank_ic_path),
            "fmb": str(fmb_path),
            "summary": str(summary_path),
            "spread_figure": str(spread_fig),
            "rank_ic_figure": str(ic_fig) if ic_fig else None,
        },
        "status": "ok" if len(sort_input) and len(rank_ic) else "needs_review",
    }
    atomic_write_json(paths.manifests / "baselines.json", manifest)
    atomic_write_text(paths.reports / "baselines_report.md", report_text(manifest, summary))
    return manifest


def report_text(manifest: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# Baselines Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Panel run ID: `{manifest['panel_run_id']}`",
        f"- Status: `{manifest['status']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| `{row['metric']}` | {row['value']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_baselines(project_root, args.panel_run_id, args.run_id)
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'baselines.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
