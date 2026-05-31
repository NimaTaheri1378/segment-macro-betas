from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, require_project_root, resolve_project_root


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


def read_manifest(project_root: Path, run_id: str) -> dict:
    path = project_root / "runs" / run_id / "manifests" / "smoke_panel.json"
    return json.loads(path.read_text(encoding="utf-8"))


def plot_hist_or_constant(ax, series: pd.Series, *, title: str, xlabel: str, color: str, default_bins: int = 12) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values.astype(float)
    if values.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        lo = float(values.min())
        hi = float(values.max())
    if values.empty:
        pass
    elif values.nunique(dropna=True) <= 1 or not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        x = float(values.iloc[0])
        width = 0.05 if x == 0 else abs(x) * 0.05
        ax.bar([x], [len(values)], width=width, color=color, edgecolor="white")
        ax.set_xlim(x - width * 3, x + width * 3)
    else:
        bins = min(default_bins, max(3, int(values.nunique(dropna=True))))
        edges = np.linspace(lo, hi, bins + 1)
        ax.hist(values, bins=edges, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)


def build_figures(project_root: Path, run_id: str) -> dict[str, str]:
    plt = setup_matplotlib()
    manifest = read_manifest(project_root, run_id)
    panel = pd.read_parquet(project_root / "data" / "interim" / run_id / "smoke_panel.parquet")
    fig_dir = ensure_within(project_root, project_root / "artifacts" / "figures_static" / run_id)
    fig_dir.mkdir(parents=True, exist_ok=True)

    checks = manifest["checks"]
    waterfall = pd.DataFrame(
        {
            "stage": ["Segment rows", "Clean segment rows", "CRSP rows", "Panel rows"],
            "rows": [
                checks.get("segment_rows_raw", 0),
                checks.get("segment_rows_after_cleaning", 0),
                checks.get("crsp_rows", 0),
                checks.get("panel_rows", 0),
            ],
        }
    )
    colors = ["#384B70", "#2D8A7D", "#C27D38", "#8E4D7B"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(waterfall["stage"], waterfall["rows"], color=colors)
    ax.set_title("Smoke Panel Sample Waterfall")
    ax.set_ylabel("Rows")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=18)
    for idx, value in enumerate(waterfall["rows"]):
        ax.text(idx, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    waterfall_path = fig_dir / "sample_waterfall.png"
    fig.savefig(waterfall_path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    plot_hist_or_constant(
        axes[0],
        panel["foreign_share"],
        title="Foreign Share",
        xlabel="Revenue share",
        color="#2D8A7D",
    )
    axes[0].set_ylabel("Firm-months")
    plot_hist_or_constant(
        axes[1],
        panel["geo_hhi"],
        title="Geography Concentration",
        xlabel="HHI",
        color="#C27D38",
    )
    fig.tight_layout()
    exposure_path = fig_dir / "exposure_distributions.png"
    fig.savefig(exposure_path)
    plt.close(fig)

    by_month = panel.groupby("date", as_index=False).agg(
        mean_foreign_share=("foreign_share", "mean"),
        mean_geo_hhi=("geo_hhi", "mean"),
        firms=("gvkey", "nunique"),
    )
    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))
    ax1.plot(by_month["date"], by_month["mean_foreign_share"], color="#2D8A7D", linewidth=2.0, label="Foreign share")
    ax1.plot(by_month["date"], by_month["mean_geo_hhi"], color="#8E4D7B", linewidth=2.0, label="Geo HHI")
    ax1.set_title("Smoke Exposure Time Series")
    ax1.set_ylabel("Cross-sectional mean")
    ax1.legend(frameon=False, loc="best")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    timeseries_path = fig_dir / "exposure_timeseries.png"
    fig.savefig(timeseries_path)
    plt.close(fig)

    outputs = {
        "sample_waterfall": str(waterfall_path),
        "exposure_distributions": str(exposure_path),
        "exposure_timeseries": str(timeseries_path),
    }
    fig_manifest = {
        "run_id": run_id,
        "source_manifest": str(project_root / "runs" / run_id / "manifests" / "smoke_panel.json"),
        "figures": outputs,
    }
    manifest_path = project_root / "runs" / run_id / "manifests" / "smoke_figures.json"
    manifest_path.write_text(json.dumps(fig_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    outputs = build_figures(project_root, args.run_id)
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
