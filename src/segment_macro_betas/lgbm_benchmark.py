from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_macro_betas.baselines import safe_quintiles, setup_matplotlib, t_stat
from segment_macro_betas.io_utils import atomic_write_json, atomic_write_parquet, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


TARGET = "next_month_excess_ret"
DEFAULT_FEATURES = [
    "foreign_share",
    "domestic_share",
    "geo_hhi",
    "geo_count",
    "top_geo_share",
    "log_segment_sales",
    "log_mktcap",
    "log_at",
    "book_to_market",
    "sales_to_assets",
    "profitability",
    "capx_to_assets",
    "rd_to_assets",
    "leverage",
    "ret",
    "excess_ret",
    "mktrf",
    "smb",
    "hml",
    "umd",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_log(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    positive = values.where(values > 0).astype("float64")
    return pd.Series(np.log(positive.to_numpy()), index=series.index)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.where(den.abs() > 0)


def build_feature_frame(panel: pd.DataFrame, *, holdout_start: str = "2026-01-01") -> tuple[pd.DataFrame, list[str]]:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna() & (df["date"] < pd.Timestamp(holdout_start))].copy()
    if TARGET not in df.columns:
        raise KeyError(f"Required target column missing: {TARGET}")

    for col in set(DEFAULT_FEATURES + [TARGET, "mktcap", "at", "ceq", "sale", "ni", "capx", "xrd", "dltt", "dlc", "segment_sales_sum"]):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "segment_sales_sum" in df:
        df["log_segment_sales"] = safe_log(df["segment_sales_sum"])
    if "mktcap" in df:
        df["log_mktcap"] = safe_log(df["mktcap"].abs())
    if "at" in df:
        df["log_at"] = safe_log(df["at"])
    if {"ceq", "mktcap"}.issubset(df.columns):
        df["book_to_market"] = safe_ratio(df["ceq"], df["mktcap"].abs())
    if {"sale", "at"}.issubset(df.columns):
        df["sales_to_assets"] = safe_ratio(df["sale"], df["at"])
    if {"ni", "at"}.issubset(df.columns):
        df["profitability"] = safe_ratio(df["ni"], df["at"])
    if {"capx", "at"}.issubset(df.columns):
        df["capx_to_assets"] = safe_ratio(df["capx"], df["at"])
    if {"xrd", "at"}.issubset(df.columns):
        df["rd_to_assets"] = safe_ratio(df["xrd"].fillna(0.0), df["at"])
    if {"dltt", "dlc", "at"}.issubset(df.columns):
        df["leverage"] = safe_ratio(df["dltt"].fillna(0.0) + df["dlc"].fillna(0.0), df["at"])

    features = [col for col in DEFAULT_FEATURES if col in df.columns]
    keep = ["gvkey", "permno", "date", TARGET] + features
    df = df[keep].replace([np.inf, -np.inf], np.nan)
    for col in [TARGET] + features:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df[df[TARGET].notna()].copy()
    return df, features


def make_yearly_folds(dates: pd.Series, *, min_train_months: int = 36, holdout_start: str = "2026-01-01") -> list[dict[str, Any]]:
    unique_dates = pd.Series(pd.to_datetime(dates, errors="coerce").dropna().unique()).sort_values()
    unique_dates = unique_dates[unique_dates < pd.Timestamp(holdout_start)]
    folds: list[dict[str, Any]] = []
    for year in sorted(unique_dates.dt.year.unique()):
        val_dates = unique_dates[unique_dates.dt.year == year]
        if val_dates.empty:
            continue
        train_dates = unique_dates[unique_dates < val_dates.min()]
        if len(train_dates) < min_train_months:
            continue
        folds.append(
            {
                "fold_year": int(year),
                "train_start": train_dates.min(),
                "train_end": train_dates.max(),
                "validation_start": val_dates.min(),
                "validation_end": val_dates.max(),
                "train_months": int(len(train_dates)),
                "validation_months": int(len(val_dates)),
            }
        )
    return folds


def monthly_rank_ic(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in predictions.groupby("date", sort=True):
        g = group[["prediction", TARGET]].dropna()
        if len(g) < 25 or g["prediction"].nunique() < 3 or g[TARGET].nunique() < 3:
            continue
        rows.append(
            {
                "date": date,
                "rank_ic": g["prediction"].rank(method="average").corr(g[TARGET].rank(method="average")),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def prediction_quintile_returns(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = predictions.copy()
    scored["prediction_quintile"] = scored.groupby("date", group_keys=False)["prediction"].apply(safe_quintiles)
    scored = scored[scored["prediction_quintile"].notna()].copy()
    scored["prediction_quintile"] = scored["prediction_quintile"].astype(int)
    quintiles = (
        scored.groupby(["date", "prediction_quintile"], as_index=False)
        .agg(mean_next_excess_ret=(TARGET, "mean"), n=("permno", "nunique"), mean_prediction=("prediction", "mean"))
        .sort_values(["date", "prediction_quintile"])
    )
    wide = quintiles.pivot(index="date", columns="prediction_quintile", values="mean_next_excess_ret")
    spread = (wide.get(5) - wide.get(1)).rename("q5_minus_q1").reset_index()
    spread["cum_q5_minus_q1"] = (1.0 + spread["q5_minus_q1"].fillna(0.0)).cumprod() - 1.0
    return quintiles, spread


def fit_predict_lgbm(
    frame: pd.DataFrame,
    features: list[str],
    folds: list[dict[str, Any]],
    *,
    n_jobs: int,
    max_train_rows_per_fold: int | None = None,
    seed: int = 137,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - exercised on environments without optional dependency
        raise RuntimeError("lightgbm is required for the benchmark; install the model optional dependencies.") from exc

    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    for fold_index, fold in enumerate(folds):
        train_mask = (frame["date"] >= fold["train_start"]) & (frame["date"] <= fold["train_end"])
        val_mask = (frame["date"] >= fold["validation_start"]) & (frame["date"] <= fold["validation_end"])
        train = frame.loc[train_mask].dropna(subset=[TARGET]).copy()
        val = frame.loc[val_mask].dropna(subset=[TARGET]).copy()
        if train.empty or val.empty:
            continue
        if max_train_rows_per_fold and len(train) > max_train_rows_per_fold:
            train = train.sample(max_train_rows_per_fold, random_state=seed + fold_index)

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.25,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        model.fit(train[features], train[TARGET])
        pred = val[["gvkey", "permno", "date", TARGET]].copy()
        pred["prediction"] = model.predict(val[features])
        pred["fold_year"] = fold["fold_year"]
        pred["train_rows"] = int(len(train))
        prediction_frames.append(pred)

        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": features,
                    "importance_gain_proxy": model.feature_importances_,
                    "fold_year": fold["fold_year"],
                }
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    importances = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return predictions, importances


def render_figures(figures_dir: Path, spread: pd.DataFrame, rank_ic: pd.DataFrame) -> dict[str, str | None]:
    plt = setup_matplotlib()
    outputs: dict[str, str | None] = {"prediction_spread": None, "rank_ic": None}
    if len(spread):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(spread["date"], spread["cum_q5_minus_q1"], color="#226F54", linewidth=2.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("LightGBM Predicted Q5 - Q1 Cumulative Return")
        ax.set_ylabel("Cumulative return")
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        path = figures_dir / "lgbm_prediction_spread_cumulative.png"
        fig.savefig(path)
        plt.close(fig)
        outputs["prediction_spread"] = str(path)
    if len(rank_ic):
        ric = rank_ic.copy()
        ric["rolling_12m"] = ric["rank_ic"].rolling(12, min_periods=3).mean()
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(ric["date"], ric["rank_ic"], color="#B56576", alpha=0.45, width=20)
        ax.plot(ric["date"], ric["rolling_12m"], color="#2F3E46", linewidth=2.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("LightGBM Monthly Rank IC")
        ax.set_ylabel("Rank IC")
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        path = figures_dir / "lgbm_rank_ic.png"
        fig.savefig(path)
        plt.close(fig)
        outputs["rank_ic"] = str(path)
    return outputs


def run_lgbm_benchmark(
    project_root: Path,
    panel_run_id: str,
    run_id: str,
    *,
    n_jobs: int = 4,
    min_train_months: int = 36,
    max_train_rows_per_fold: int | None = None,
) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    panel_path = ensure_within(project_root, project_root / "data" / "interim" / panel_run_id / "monthly_panel.parquet")
    tables_dir = ensure_within(project_root, project_root / "artifacts" / "tables" / run_id)
    figures_dir = ensure_within(project_root, project_root / "artifacts" / "figures_static" / run_id)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    frame, features = build_feature_frame(panel)
    folds = make_yearly_folds(frame["date"], min_train_months=min_train_months)
    predictions, importances = fit_predict_lgbm(
        frame,
        features,
        folds,
        n_jobs=n_jobs,
        max_train_rows_per_fold=max_train_rows_per_fold,
    )
    rank_ic = monthly_rank_ic(predictions) if len(predictions) else pd.DataFrame()
    quintiles, spread = prediction_quintile_returns(predictions) if len(predictions) else (pd.DataFrame(), pd.DataFrame())
    if len(predictions):
        prediction_counts = predictions.groupby("fold_year", as_index=False).agg(
            prediction_rows=("prediction", "size"),
            train_rows=("train_rows", "max"),
        )
    else:
        prediction_counts = pd.DataFrame()
    if len(predictions) and len(rank_ic):
        rank_ic_folds = rank_ic.merge(predictions[["date", "fold_year"]].drop_duplicates(), on="date", how="left")
        fold_rank_ic = rank_ic_folds.groupby("fold_year", as_index=False).agg(
            rank_ic_months=("rank_ic", "size"),
            mean_rank_ic=("rank_ic", "mean"),
        )
        fold_metrics = prediction_counts.merge(fold_rank_ic, on="fold_year", how="left")
    else:
        fold_metrics = prediction_counts
    feature_importance = (
        importances.groupby("feature", as_index=False)
        .agg(mean_importance=("importance_gain_proxy", "mean"), folds=("fold_year", "nunique"))
        .sort_values("mean_importance", ascending=False)
        if len(importances)
        else pd.DataFrame()
    )
    summary = pd.DataFrame(
        [
            {"metric": "feature_rows", "value": int(len(frame))},
            {"metric": "prediction_rows", "value": int(len(predictions))},
            {"metric": "features", "value": int(len(features))},
            {"metric": "folds", "value": int(len(folds))},
            {"metric": "rank_ic_months", "value": int(len(rank_ic))},
            {"metric": "mean_rank_ic", "value": rank_ic["rank_ic"].mean() if len(rank_ic) else None},
            {"metric": "t_rank_ic", "value": t_stat(rank_ic["rank_ic"]) if len(rank_ic) else None},
            {"metric": "mean_q5_minus_q1", "value": spread["q5_minus_q1"].mean() if len(spread) else None},
            {"metric": "t_q5_minus_q1", "value": t_stat(spread["q5_minus_q1"]) if len(spread) else None},
        ]
    )

    prediction_path = tables_dir / "lgbm_predictions.parquet"
    fold_metrics_path = tables_dir / "lgbm_fold_metrics.csv"
    rank_ic_path = tables_dir / "lgbm_rank_ic.csv"
    quintiles_path = tables_dir / "lgbm_prediction_quintile_returns.csv"
    spread_path = tables_dir / "lgbm_prediction_spread.csv"
    feature_importance_path = tables_dir / "lgbm_feature_importance.csv"
    summary_path = tables_dir / "lgbm_summary.csv"
    atomic_write_parquet(predictions, prediction_path)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    rank_ic.to_csv(rank_ic_path, index=False)
    quintiles.to_csv(quintiles_path, index=False)
    spread.to_csv(spread_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    summary.to_csv(summary_path, index=False)
    figure_outputs = render_figures(figures_dir, spread, rank_ic)

    status = "ok" if len(predictions) and len(rank_ic) and len(features) else "needs_review"
    manifest = {
        "run_id": run_id,
        "panel_run_id": panel_run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "model": "lightgbm_regressor_expanding_yearly",
        "features": features,
        "validation": {
            "scheme": "expanding_train_yearly_validation",
            "min_train_months": int(min_train_months),
            "holdout_start": "2026-01-01",
            "folds": [
                {
                    key: (str(value.date()) if hasattr(value, "date") else value)
                    for key, value in fold.items()
                }
                for fold in folds
            ],
        },
        "inputs": {"panel": str(panel_path), "panel_rows": int(len(panel)), "feature_rows": int(len(frame))},
        "checks": {
            "prediction_rows": int(len(predictions)),
            "rank_ic_months": int(len(rank_ic)),
            "quintile_months": int(spread["date"].nunique()) if len(spread) else 0,
            "mean_rank_ic": float(rank_ic["rank_ic"].mean()) if len(rank_ic) else None,
            "t_rank_ic": t_stat(rank_ic["rank_ic"]) if len(rank_ic) else None,
            "mean_q5_minus_q1": float(spread["q5_minus_q1"].mean()) if len(spread) else None,
            "t_q5_minus_q1": t_stat(spread["q5_minus_q1"]) if len(spread) else None,
        },
        "outputs": {
            "predictions": str(prediction_path),
            "fold_metrics": str(fold_metrics_path),
            "rank_ic": str(rank_ic_path),
            "quintile_returns": str(quintiles_path),
            "spread": str(spread_path),
            "feature_importance": str(feature_importance_path),
            "summary": str(summary_path),
            **figure_outputs,
        },
        "status": status,
    }
    atomic_write_json(paths.manifests / "lgbm_benchmark.json", manifest)
    atomic_write_text(paths.reports / "lgbm_benchmark_report.md", report_text(manifest, summary, feature_importance))
    return manifest


def report_text(manifest: dict[str, Any], summary: pd.DataFrame, feature_importance: pd.DataFrame) -> str:
    lines = [
        "# LightGBM Benchmark Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Panel run ID: `{manifest['panel_run_id']}`",
        f"- Status: `{manifest['status']}`",
        f"- Validation: `{manifest['validation']['scheme']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(f"| `{row['metric']}` | {row['value']} |")
    if len(feature_importance):
        lines.extend(["", "## Top Features", "", "| Feature | Mean importance |", "|---|---:|"])
        for row in feature_importance.head(12).to_dict(orient="records"):
            lines.append(f"| `{row['feature']}` | {row['mean_importance']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--min-train-months", type=int, default=36)
    parser.add_argument("--max-train-rows-per-fold", type=int, default=None)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_lgbm_benchmark(
        project_root,
        args.panel_run_id,
        args.run_id,
        n_jobs=args.n_jobs,
        min_train_months=args.min_train_months,
        max_train_rows_per_fold=args.max_train_rows_per_fold,
    )
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'lgbm_benchmark.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
