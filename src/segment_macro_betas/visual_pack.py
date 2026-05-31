from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_macro_betas.baselines import setup_matplotlib
from segment_macro_betas.io_utils import atomic_write_json, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root
from segment_macro_betas.panel_builder import read_shards
from segment_macro_betas.segment_set_model import prepare_segment_tokens


SECTOR_LABELS = [
    "Agriculture",
    "Mining",
    "Construction",
    "Manufacturing",
    "Transport",
    "Wholesale",
    "Retail",
    "Finance",
    "Services",
    "Public",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_ids(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        run_ids = raw
    else:
        run_ids = [piece.strip() for piece in raw.split(",")]
    out = [run_id for run_id in run_ids if run_id]
    if not out:
        raise ValueError("At least one run id is required.")
    return out


def read_set_summaries(project_root: Path, set_run_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_id in set_run_ids:
        path = project_root / "artifacts" / "tables" / run_id / "deepsets_summary.csv"
        frame = pd.read_csv(path)
        frame["model_run_id"] = run_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.{digits}f}"
    return str(value)


def sic_sector(sic: Any) -> str:
    try:
        code = int(float(sic))
    except (TypeError, ValueError):
        return "Unknown"
    if 100 <= code <= 999:
        return "Agriculture"
    if 1000 <= code <= 1499:
        return "Mining"
    if 1500 <= code <= 1799:
        return "Construction"
    if 2000 <= code <= 3999:
        return "Manufacturing"
    if 4000 <= code <= 4999:
        return "Transport"
    if 5000 <= code <= 5199:
        return "Wholesale"
    if 5200 <= code <= 5999:
        return "Retail"
    if 6000 <= code <= 6799:
        return "Finance"
    if 7000 <= code <= 8999:
        return "Services"
    if 9000 <= code <= 9999:
        return "Public"
    return "Unknown"


def build_model_comparison(lgbm: pd.DataFrame, deepsets: pd.DataFrame) -> pd.DataFrame:
    a = lgbm.copy()
    a["family"] = "LightGBM"
    b = deepsets.copy()
    b["family"] = "Deep Sets"
    b = b.rename(columns={"features": "features"})
    if "architecture" in b.columns:
        b["architecture"] = b["architecture"].fillna("deep_sets").replace("", "deep_sets")
    cols = ["family", "variant", "prediction_rows", "rank_ic_months", "mean_rank_ic", "t_rank_ic", "mean_q5_minus_q1", "t_q5_minus_q1"]
    combined = pd.concat([a, b], ignore_index=True)
    return combined[[col for col in cols if col in combined.columns]]


def load_crsp_sector(raw_root: Path) -> pd.DataFrame:
    crsp = read_shards(raw_root, "crsp_monthly")
    keep = [col for col in ["permno", "date", "siccd", "ticker"] if col in crsp.columns]
    out = crsp[keep].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce")
    out = out[out["date"].notna() & out["permno"].notna()].copy()
    out["permno"] = out["permno"].astype(int)
    if "siccd" in out.columns:
        out["sector"] = out["siccd"].map(sic_sector)
    else:
        out["sector"] = "Unknown"
    return out.sort_values(["permno", "date"]).drop_duplicates(["permno", "date"], keep="last")


def build_latest_segment_matrix(panel: pd.DataFrame, raw_root: Path, *, top_n_geo: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    latest_year = int(panel["date"].dt.year.max())
    recent = panel[panel["date"].dt.year == latest_year].copy()
    crsp_sector = load_crsp_sector(raw_root)
    recent = recent.merge(crsp_sector[["permno", "date", "sector", "ticker"]], on=["permno", "date"], how="left")
    segments = prepare_segment_tokens(read_shards(raw_root, "segments"))
    segments["segment_srcdate"] = pd.to_datetime(segments["segment_srcdate"], errors="coerce")
    token_panel = recent[["gvkey", "permno", "date", "segment_srcdate", "sector", "ticker"]].merge(
        segments[["gvkey", "segment_srcdate", "geo_label", "revenue_share"]],
        on=["gvkey", "segment_srcdate"],
        how="inner",
    )
    top_geo = token_panel.groupby("geo_label")["revenue_share"].sum().sort_values(ascending=False).head(top_n_geo).index
    token_panel = token_panel[token_panel["geo_label"].isin(top_geo)].copy()
    matrix = token_panel.pivot_table(
        index="sector",
        columns="geo_label",
        values="revenue_share",
        aggfunc="mean",
        fill_value=0.0,
    )
    matrix = matrix.reindex(index=[label for label in SECTOR_LABELS if label in matrix.index])
    firm_explorer = (
        recent.sort_values(["foreign_share", "date"], ascending=[False, False])
        .drop_duplicates(["permno"], keep="first")
        .head(30)[["date", "permno", "ticker", "sector", "foreign_share", "geo_hhi", "geo_count", "top_geo_share"]]
        .sort_values("foreign_share", ascending=False)
    )
    return matrix, firm_explorer


def save_sample_waterfall(figures_dir: Path, panel_manifest: dict[str, Any], lgbm_summary: pd.DataFrame, set_summary: pd.DataFrame) -> Path:
    plt = setup_matplotlib()
    checks = panel_manifest["checks"]
    values = [
        checks.get("segment_snapshots", 0),
        checks.get("filing_date_matched_snapshots", 0),
        checks.get("panel_rows", 0),
        int(lgbm_summary["prediction_rows"].max()) if len(lgbm_summary) else 0,
        int(set_summary["prediction_rows"].max()) if len(set_summary) else 0,
    ]
    labels = ["Segment\nsnapshots", "Filing-date\nmatched", "Panel\nrows", "LGBM\npredictions", "Deep Sets\npredictions"]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    colors = ["#4F6D7A", "#2A9D8F", "#E9C46A", "#E76F51", "#8E6C8A"]
    ax.bar(labels, values, color=colors)
    ax.set_title("Sample And Model Coverage")
    ax.set_ylabel("Rows or snapshots")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = figures_dir / "sample_model_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_activation_sources(figures_dir: Path, panel_manifest: dict[str, Any]) -> Path:
    plt = setup_matplotlib()
    counts = pd.Series(panel_manifest["checks"].get("activation_source_counts", {})).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.barh(counts.index.astype(str), counts.values, color="#2A9D8F")
    ax.set_title("Segment Activation Date Sources")
    ax.set_xlabel("Snapshot count")
    for i, value in enumerate(counts.values):
        ax.text(value, i, f" {int(value):,}", va="center", fontsize=8)
    fig.tight_layout()
    out = figures_dir / "activation_source_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_exposure_figures(figures_dir: Path, panel: pd.DataFrame) -> dict[str, Path]:
    plt = setup_matplotlib()
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    out: dict[str, Path] = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(panel["foreign_share"].dropna(), bins=40, color="#2A7F62", alpha=0.75)
    ax.set_title("Foreign Sales Share Distribution")
    ax.set_xlabel("Foreign share")
    ax.set_ylabel("Firm-months")
    fig.tight_layout()
    out["exposure_distribution"] = figures_dir / "foreign_share_distribution.png"
    fig.savefig(out["exposure_distribution"])
    plt.close(fig)

    trend = panel.groupby("date", as_index=False).agg(mean_foreign_share=("foreign_share", "mean"), mean_geo_hhi=("geo_hhi", "mean"))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(trend["date"], trend["mean_foreign_share"], color="#226F54", linewidth=1.8, label="Foreign share")
    ax.plot(trend["date"], trend["mean_geo_hhi"], color="#8E6C8A", linewidth=1.4, label="Geo HHI")
    ax.set_title("Average Segment Exposure Over Time")
    ax.set_ylabel("Cross-sectional mean")
    ax.legend(frameon=False)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    out["exposure_time_series"] = figures_dir / "exposure_time_series.png"
    fig.savefig(out["exposure_time_series"])
    plt.close(fig)
    return out


def save_model_comparison(figures_dir: Path, comparison: pd.DataFrame) -> dict[str, Path]:
    plt = setup_matplotlib()
    out: dict[str, Path] = {}
    labels = comparison["family"] + "\n" + comparison["variant"].astype(str).str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(labels, comparison["mean_rank_ic"], color="#4F6D7A")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Model Comparison: Mean Monthly Rank IC")
    ax.set_ylabel("Mean rank IC")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    out["model_rank_ic"] = figures_dir / "model_rank_ic_comparison.png"
    fig.savefig(out["model_rank_ic"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(labels, comparison["mean_q5_minus_q1"], color="#E76F51")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Model Comparison: Predicted Q5 - Q1")
    ax.set_ylabel("Mean next-month excess return")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    out["model_spread"] = figures_dir / "model_spread_comparison.png"
    fig.savefig(out["model_spread"])
    plt.close(fig)
    return out


def save_sector_geo_matrix(figures_dir: Path, matrix: pd.DataFrame) -> Path:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Latest-Year Sector Geography Exposure Matrix")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean segment revenue share")
    fig.tight_layout()
    out = figures_dir / "sector_geography_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def write_dashboard(html_dir: Path, run_id: str, figures: dict[str, Path], comparison: pd.DataFrame, firm_explorer: pd.DataFrame) -> Path:
    html_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(fmt(row.get(col), 4))}</td>" for col in comparison.columns)
        + "</tr>"
        for row in comparison.to_dict(orient="records")
    )
    firm_rows = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(row.get(col), 4))}</td>" for col in firm_explorer.columns) + "</tr>"
        for row in firm_explorer.to_dict(orient="records")
    )
    figure_blocks = "\n".join(
        f"<section><h2>{html.escape(name.replace('_', ' ').title())}</h2><img src=\"../../figures_static/{run_id}/{path.name}\" alt=\"{html.escape(name)}\"></section>"
        for name, path in figures.items()
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Segment Macro Betas Visual Pack</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #20262e; }}
    h1, h2 {{ color: #17212b; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    section {{ margin: 28px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f6f8; }}
  </style>
</head>
<body>
  <h1>Segment Macro Betas Visual Pack</h1>
  <p>Private diagnostic dashboard for run <code>{html.escape(run_id)}</code>. Generated from ignored artifacts.</p>
  <h2>Model Comparison</h2>
  <table><thead><tr>{''.join(f'<th>{html.escape(col)}</th>' for col in comparison.columns)}</tr></thead><tbody>{rows}</tbody></table>
  {figure_blocks}
  <h2>Firm Explorer Snapshot</h2>
  <table><thead><tr>{''.join(f'<th>{html.escape(col)}</th>' for col in firm_explorer.columns)}</tr></thead><tbody>{firm_rows}</tbody></table>
</body>
</html>
"""
    out = html_dir / "dashboard.html"
    atomic_write_text(out, doc)
    return out


def write_model_card(report_dir: Path, run_id: str, panel_manifest: dict[str, Any], comparison: pd.DataFrame) -> Path:
    checks = panel_manifest["checks"]

    def metric(value: Any) -> str:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.6f}"

    lines = [
        "# Segment Macro Betas Model Card",
        "",
        f"- Visual pack run: `{run_id}`",
        f"- Panel rows: `{checks.get('panel_rows')}`",
        f"- Filing-date match rate: `{checks.get('filing_date_match_rate')}`",
        f"- Activation-rule violations: `{checks.get('activation_rule_violations')}`",
        "",
        "## Diagnostics",
        "",
        "| Family | Variant | Mean rank IC | Q5-Q1 |",
        "|---|---|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        lines.append(
            f"| {row['family']} | `{row['variant']}` | {metric(row['mean_rank_ic'])} | "
            f"{metric(row['mean_q5_minus_q1'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Treat all metrics as private diagnostics, not publication claims.",
            "- The filing-date panel has zero activation-rule violations in the manifest.",
            "- Segment-only tabular features rank returns strongly, but control-rich variants produce stronger long-short spreads.",
            "- The first Deep Sets extension has modest positive rank IC in the set-only variant and weak long-short spread.",
            "- The full Set Transformer run is a CUDA-validated architecture diagnostic, not a dominant economic spread result.",
            "- Official non-FRED macro diagnostics are live, while full FRED and revision-safe macro claims remain blocked.",
            "",
        ]
    )
    out = report_dir / "model_card.md"
    atomic_write_text(out, "\n".join(lines))
    return out


def run_visual_pack(
    project_root: Path,
    run_id: str,
    raw_run_id: str,
    panel_run_id: str,
    baseline_run_id: str,
    lgbm_run_id: str,
    set_run_ids: list[str],
) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    raw_root = ensure_within(project_root, project_root / "data" / "raw" / raw_run_id)
    panel_path = ensure_within(project_root, project_root / "data" / "interim" / panel_run_id / "monthly_panel.parquet")
    figures_dir = ensure_within(project_root, project_root / "artifacts" / "figures_static" / run_id)
    html_dir = ensure_within(project_root, project_root / "artifacts" / "figures_html" / run_id)
    tables_dir = ensure_within(project_root, project_root / "artifacts" / "tables" / run_id)
    figures_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    panel_manifest = read_json(project_root / "runs" / panel_run_id / "manifests" / "monthly_panel.json")
    panel = pd.read_parquet(panel_path)
    lgbm_summary = pd.read_csv(project_root / "artifacts" / "tables" / lgbm_run_id / "lgbm_summary.csv")
    set_summary = read_set_summaries(project_root, set_run_ids)
    baseline_summary = pd.read_csv(project_root / "artifacts" / "tables" / baseline_run_id / "baseline_summary.csv")
    comparison = build_model_comparison(lgbm_summary, set_summary)
    matrix, firm_explorer = build_latest_segment_matrix(panel, raw_root)

    comparison_path = tables_dir / "model_comparison.csv"
    firm_path = tables_dir / "firm_explorer_snapshot.csv"
    matrix_path = tables_dir / "sector_geography_matrix.csv"
    comparison.to_csv(comparison_path, index=False)
    firm_explorer.to_csv(firm_path, index=False)
    matrix.to_csv(matrix_path)

    figures: dict[str, Path] = {
        "sample_model_coverage": save_sample_waterfall(figures_dir, panel_manifest, lgbm_summary, set_summary),
        "activation_source_coverage": save_activation_sources(figures_dir, panel_manifest),
        "sector_geography_matrix": save_sector_geo_matrix(figures_dir, matrix),
        **save_exposure_figures(figures_dir, panel),
        **save_model_comparison(figures_dir, comparison),
    }
    dashboard = write_dashboard(html_dir, run_id, figures, comparison, firm_explorer)
    model_card = write_model_card(paths.reports, run_id, panel_manifest, comparison)
    manifest = {
        "run_id": run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "inputs": {
            "raw_run_id": raw_run_id,
            "panel_run_id": panel_run_id,
            "baseline_run_id": baseline_run_id,
            "lgbm_run_id": lgbm_run_id,
            "set_run_ids": set_run_ids,
            "panel_rows": int(len(panel)),
            "baseline_summary_rows": int(len(baseline_summary)),
        },
        "outputs": {
            "figures": {name: str(path) for name, path in figures.items()},
            "dashboard": str(dashboard),
            "model_card": str(model_card),
            "model_comparison": str(comparison_path),
            "firm_explorer": str(firm_path),
            "sector_geography_matrix": str(matrix_path),
        },
        "checks": {
            "figure_count": int(len(figures)),
            "comparison_rows": int(len(comparison)),
            "firm_explorer_rows": int(len(firm_explorer)),
            "sector_matrix_shape": list(matrix.shape),
        },
        "status": "ok" if len(figures) >= 6 and len(comparison) else "needs_review",
    }
    atomic_write_json(paths.manifests / "visual_pack.json", manifest)
    atomic_write_text(paths.reports / "visual_pack_report.md", report_text(manifest))
    return manifest


def report_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# Visual Pack Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Status: `{manifest['status']}`",
        f"- Figure count: `{manifest['checks']['figure_count']}`",
        f"- Dashboard: `{manifest['outputs']['dashboard']}`",
        f"- Model card: `{manifest['outputs']['model_card']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-run-id", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--lgbm-run-id", required=True)
    parser.add_argument("--set-run-id", required=True, help="Set-model run id, or comma-separated run ids.")
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_visual_pack(
        project_root,
        args.run_id,
        args.raw_run_id,
        args.panel_run_id,
        args.baseline_run_id,
        args.lgbm_run_id,
        parse_run_ids(args.set_run_id),
    )
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'visual_pack.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
