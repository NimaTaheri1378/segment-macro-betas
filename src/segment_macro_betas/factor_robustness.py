from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_macro_betas.baselines import safe_quintiles, setup_matplotlib, t_stat
from segment_macro_betas.io_utils import atomic_write_json, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


TARGET = "next_month_excess_ret"
FACTOR_COLUMNS = ["mktrf", "smb", "hml", "umd"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_run_specs(raw: str) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Run spec must be family:run_id, got {item!r}")
        family, run_id = item.split(":", 1)
        family = family.strip().lower()
        run_id = run_id.strip()
        if family not in {"lgbm", "deepsets"}:
            raise ValueError(f"Unknown model family in run spec: {family}")
        if not run_id:
            raise ValueError("Run id cannot be empty.")
        specs.append((family, run_id))
    if not specs:
        raise ValueError("At least one model run spec is required.")
    return specs


def next_month_end(dates: pd.Series) -> pd.Series:
    return pd.to_datetime(dates, errors="coerce") + pd.offsets.MonthEnd(1)


def load_model_predictions(tables_root: Path, specs: list[tuple[str, str]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family, run_id in specs:
        run_dir = tables_root / run_id
        pattern = "lgbm_*_predictions.parquet" if family == "lgbm" else "deepsets_*_predictions.parquet"
        for path in sorted(run_dir.glob(pattern)):
            variant = path.name.removeprefix("lgbm_").removeprefix("deepsets_").removesuffix("_predictions.parquet")
            df = pd.read_parquet(path)
            required = {"gvkey", "permno", "date", "prediction", TARGET}
            missing = required.difference(df.columns)
            if missing:
                raise KeyError(f"{path} missing required columns: {sorted(missing)}")
            keep = df[["gvkey", "permno", "date", "prediction", TARGET]].copy()
            keep["model_family"] = family
            keep["model_run_id"] = run_id
            keep["variant"] = variant
            frames.append(keep)
    if not frames:
        raise FileNotFoundError(f"No prediction files found under {tables_root} for specs {specs}")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce")
    out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    return out.dropna(subset=["date", "permno", "prediction", TARGET]).copy()


def prepare_factor_returns(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    available = [col for col in FACTOR_COLUMNS if col in df.columns]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    factors = df[["date"] + available].dropna(subset=["date"]).drop_duplicates("date", keep="last").copy()
    return factors.sort_values("date")


def prediction_quintile_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.copy()
    group_cols = ["model_family", "model_run_id", "variant", "date"]
    scored["prediction_quintile"] = scored.groupby(group_cols, group_keys=False)["prediction"].apply(safe_quintiles)
    scored = scored[scored["prediction_quintile"].notna()].copy()
    scored["prediction_quintile"] = scored["prediction_quintile"].astype(int)
    return scored


def compute_spreads(scored: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_family", "model_run_id", "variant", "date", "prediction_quintile"]
    quintiles = (
        scored.groupby(group_cols, as_index=False)
        .agg(mean_next_excess_ret=(TARGET, "mean"), n=("permno", "nunique"), mean_prediction=("prediction", "mean"))
        .sort_values(group_cols)
    )
    rows: list[dict[str, Any]] = []
    for keys, group in quintiles.groupby(["model_family", "model_run_id", "variant", "date"], sort=False):
        q = group.set_index("prediction_quintile")
        if 1 not in q.index or 5 not in q.index:
            continue
        rows.append(
            {
                "model_family": keys[0],
                "model_run_id": keys[1],
                "variant": keys[2],
                "date": keys[3],
                "q5_ret": float(q.loc[5, "mean_next_excess_ret"]),
                "q1_ret": float(q.loc[1, "mean_next_excess_ret"]),
                "q5_minus_q1": float(q.loc[5, "mean_next_excess_ret"] - q.loc[1, "mean_next_excess_ret"]),
                "q5_n": int(q.loc[5, "n"]),
                "q1_n": int(q.loc[1, "n"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_family", "variant", "date"])


def compute_turnover(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in scored.groupby(["model_family", "model_run_id", "variant"], sort=False):
        prev_long: set[int] | None = None
        prev_short: set[int] | None = None
        for date, dated in group.sort_values("date").groupby("date", sort=True):
            long_names = set(dated.loc[dated["prediction_quintile"] == 5, "permno"].astype(int))
            short_names = set(dated.loc[dated["prediction_quintile"] == 1, "permno"].astype(int))
            if prev_long is None or prev_short is None:
                long_turnover = np.nan
                short_turnover = np.nan
            else:
                long_turnover = 1.0 - len(long_names & prev_long) / max(len(long_names), 1)
                short_turnover = 1.0 - len(short_names & prev_short) / max(len(short_names), 1)
            if pd.isna(long_turnover) and pd.isna(short_turnover):
                long_short_turnover = np.nan
            else:
                long_short_turnover = np.nanmean([long_turnover, short_turnover])
            rows.append(
                {
                    "model_family": keys[0],
                    "model_run_id": keys[1],
                    "variant": keys[2],
                    "date": date,
                    "long_turnover": long_turnover,
                    "short_turnover": short_turnover,
                    "long_short_turnover": long_short_turnover,
                    "long_n": len(long_names),
                    "short_n": len(short_names),
                }
            )
            prev_long = long_names
            prev_short = short_names
    return pd.DataFrame(rows)


def newey_west_covariance(x: np.ndarray, residuals: np.ndarray, lag: int) -> np.ndarray:
    n = x.shape[0]
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    scores = x * residuals[:, None]
    meat += scores.T @ scores
    for ell in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - ell / (lag + 1.0)
        gamma = scores[ell:].T @ scores[:-ell]
        meat += weight * (gamma + gamma.T)
    return bread @ meat @ bread


def factor_alpha(y: pd.Series, factors: pd.DataFrame, *, nw_lag: int) -> dict[str, Any]:
    data = pd.concat([pd.to_numeric(y, errors="coerce").rename("y"), factors], axis=1).dropna()
    if len(data) < max(12, factors.shape[1] + 3):
        return {"alpha": None, "alpha_t": None, "n_months": int(len(data)), "r2": None}
    x = np.column_stack([np.ones(len(data)), data[factors.columns].to_numpy(dtype=float)])
    y_arr = data["y"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(x, y_arr, rcond=None)[0]
    residuals = y_arr - x @ beta
    cov = newey_west_covariance(x, residuals, nw_lag)
    se_alpha = float(np.sqrt(cov[0, 0])) if cov[0, 0] >= 0 else np.nan
    t_alpha = float(beta[0] / se_alpha) if se_alpha and np.isfinite(se_alpha) else None
    sst = float(((y_arr - y_arr.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / sst if sst > 0 else None
    out = {
        "alpha": float(beta[0]),
        "alpha_t": t_alpha,
        "n_months": int(len(data)),
        "r2": r2,
    }
    for name, value in zip(factors.columns, beta[1:]):
        out[f"beta_{name}"] = float(value)
    return out


def summarize_factor_robustness(predictions: pd.DataFrame, factors: pd.DataFrame, *, cost_bps: float, nw_lag: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = prediction_quintile_panel(predictions)
    spreads = compute_spreads(scored)
    turnover = compute_turnover(scored)
    spreads = spreads.merge(turnover[["model_family", "model_run_id", "variant", "date", "long_short_turnover"]], on=["model_family", "model_run_id", "variant", "date"], how="left")
    spreads["cost_per_turnover"] = float(cost_bps) / 10000.0
    spreads["net_q5_minus_q1"] = spreads["q5_minus_q1"] - spreads["cost_per_turnover"] * spreads["long_short_turnover"].fillna(0.0)
    spreads["factor_date"] = next_month_end(spreads["date"])
    available_factors = [col for col in FACTOR_COLUMNS if col in factors.columns]
    merged = spreads.merge(factors[["date"] + available_factors], left_on="factor_date", right_on="date", how="left", suffixes=("", "_factor"))
    summary_rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(["model_family", "model_run_id", "variant"], sort=False):
        factor_frame = group[available_factors]
        gross_alpha = factor_alpha(group["q5_minus_q1"], factor_frame, nw_lag=nw_lag)
        net_alpha = factor_alpha(group["net_q5_minus_q1"], factor_frame, nw_lag=nw_lag)
        summary_rows.append(
            {
                "model_family": keys[0],
                "model_run_id": keys[1],
                "variant": keys[2],
                "months": int(group["q5_minus_q1"].notna().sum()),
                "mean_gross_q5_minus_q1": float(group["q5_minus_q1"].mean()),
                "t_gross_q5_minus_q1": t_stat(group["q5_minus_q1"]),
                "mean_net_q5_minus_q1": float(group["net_q5_minus_q1"].mean()),
                "t_net_q5_minus_q1": t_stat(group["net_q5_minus_q1"]),
                "mean_turnover": float(group["long_short_turnover"].mean()),
                "gross_alpha": gross_alpha["alpha"],
                "gross_alpha_t": gross_alpha["alpha_t"],
                "net_alpha": net_alpha["alpha"],
                "net_alpha_t": net_alpha["alpha_t"],
                "alpha_months": gross_alpha["n_months"],
                "factor_r2": gross_alpha["r2"],
            }
        )
    return pd.DataFrame(summary_rows), merged, turnover


def render_alpha_figure(figures_dir: Path, summary: pd.DataFrame) -> str | None:
    if summary.empty:
        return None
    plt = setup_matplotlib()
    plot = summary.copy()
    plot["label"] = plot["model_family"] + ":" + plot["variant"]
    plot = plot.sort_values("gross_alpha", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.35 * len(plot))))
    ax.barh(plot["label"], plot["gross_alpha"], color="#386FA4", alpha=0.78)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Gross Factor Alpha by Model Variant")
    ax.set_xlabel("Monthly alpha")
    fig.tight_layout()
    path = figures_dir / "factor_alpha_by_variant.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def report_text(manifest: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# Factor Robustness Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Status: `{manifest['status']}`",
        f"- Panel run ID: `{manifest['panel_run_id']}`",
        f"- Model runs: `{', '.join(manifest['model_run_specs'])}`",
        f"- Cost assumption: `{manifest['parameters']['cost_bps']} bps per one-way turnover`",
        "",
        "## Summary",
        "",
        "| Model | Variant | Gross alpha | Gross alpha t | Net alpha | Net alpha t | Mean turnover |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| `{row['model_family']}` | `{row['variant']}` | {row['gross_alpha']} | {row['gross_alpha_t']} | "
            f"{row['net_alpha']} | {row['net_alpha_t']} | {row['mean_turnover']} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_factor_robustness(
    project_root: Path,
    panel_run_id: str,
    run_id: str,
    model_run_specs: list[tuple[str, str]],
    *,
    cost_bps: float,
    nw_lag: int,
) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    panel_path = ensure_within(project_root, project_root / "data" / "interim" / panel_run_id / "monthly_panel.parquet")
    tables_root = ensure_within(project_root, project_root / "artifacts" / "tables")
    tables_dir = ensure_within(project_root, tables_root / run_id)
    figures_dir = ensure_within(project_root, project_root / "artifacts" / "figures_static" / run_id)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    factors = prepare_factor_returns(panel)
    predictions = load_model_predictions(tables_root, model_run_specs)
    summary, spreads, turnover = summarize_factor_robustness(predictions, factors, cost_bps=cost_bps, nw_lag=nw_lag)

    summary_path = tables_dir / "factor_robustness_summary.csv"
    spreads_path = tables_dir / "factor_robustness_spreads.csv"
    turnover_path = tables_dir / "factor_robustness_turnover.csv"
    summary.to_csv(summary_path, index=False)
    spreads.to_csv(spreads_path, index=False)
    turnover.to_csv(turnover_path, index=False)
    alpha_figure = render_alpha_figure(figures_dir, summary)

    manifest = {
        "run_id": run_id,
        "panel_run_id": panel_run_id,
        "model_run_specs": [f"{family}:{model_run_id}" for family, model_run_id in model_run_specs],
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "parameters": {"cost_bps": float(cost_bps), "nw_lag": int(nw_lag), "factor_columns": [col for col in FACTOR_COLUMNS if col in factors.columns]},
        "inputs": {"panel": str(panel_path), "panel_rows": int(len(panel)), "prediction_rows": int(len(predictions))},
        "checks": {
            "variants": int(len(summary)),
            "spread_month_rows": int(len(spreads)),
            "turnover_rows": int(len(turnover)),
            "factor_months": int(factors["date"].nunique()) if len(factors) else 0,
        },
        "outputs": {
            "summary": str(summary_path),
            "spreads": str(spreads_path),
            "turnover": str(turnover_path),
            "alpha_figure": alpha_figure,
        },
        "status": "ok" if len(summary) else "needs_review",
    }
    atomic_write_json(paths.manifests / "factor_robustness.json", manifest)
    atomic_write_text(paths.reports / "factor_robustness_report.md", report_text(manifest, summary))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-runs", required=True, help="Comma-separated family:run_id entries, e.g. lgbm:run1,deepsets:run2")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--nw-lag", type=int, default=6)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_factor_robustness(
        project_root,
        args.panel_run_id,
        args.run_id,
        parse_run_specs(args.model_runs),
        cost_bps=args.cost_bps,
        nw_lag=args.nw_lag,
    )
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'factor_robustness.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
