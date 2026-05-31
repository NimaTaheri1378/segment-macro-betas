from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from segment_macro_betas.baselines import setup_matplotlib, t_stat
from segment_macro_betas.io_utils import atomic_write_json, atomic_write_parquet, atomic_write_text
from segment_macro_betas.lgbm_benchmark import (
    FIRM_FEATURES,
    MARKET_FEATURES,
    RETURN_FEATURES,
    TARGET,
    make_yearly_folds,
    monthly_rank_ic,
    prediction_quintile_returns,
    safe_log,
    safe_ratio,
)
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root
from segment_macro_betas.panel_builder import read_shards


CONTROL_FEATURES = FIRM_FEATURES + RETURN_FEATURES + MARKET_FEATURES
SET_VARIANTS = {
    "set_only": [],
    "set_plus_controls": CONTROL_FEATURES,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_variants(raw: str | None) -> list[str]:
    if not raw:
        return ["set_only", "set_plus_controls"]
    variants = [piece.strip() for piece in raw.split(",") if piece.strip()]
    unknown = [variant for variant in variants if variant not in SET_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown set-model variant(s): {', '.join(unknown)}")
    return variants


def normalize_geo_label(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    label = str(value).strip().upper()
    return label if label else "UNKNOWN"


def prepare_segment_tokens(segments: pd.DataFrame) -> pd.DataFrame:
    df = segments.copy()
    df["gvkey"] = df["gvkey"].astype(str)
    df["segment_srcdate"] = pd.to_datetime(df["srcdate"], errors="coerce")
    geo = df["gareag"].where(df["gareag"].notna() & (df["gareag"].astype(str).str.len() > 0), df["gareat"])
    df["geo_label"] = geo.map(normalize_geo_label)
    sales = pd.to_numeric(df["sales"], errors="coerce")
    revts = pd.to_numeric(df["revts"], errors="coerce")
    ias = pd.to_numeric(df["ias"], errors="coerce")
    df["segment_sales"] = sales.where(sales.notna(), revts).where(lambda s: s.notna(), ias)
    df = df[df["segment_srcdate"].notna() & df["segment_sales"].notna() & (df["segment_sales"] > 0)].copy()
    grouped = (
        df.groupby(["gvkey", "segment_srcdate", "geo_label"], as_index=False, dropna=False)
        .agg(segment_sales=("segment_sales", "sum"), segment_rows=("sid", "count"))
        .sort_values(["gvkey", "segment_srcdate", "segment_sales"], ascending=[True, True, False])
    )
    grouped["snapshot_sales"] = grouped.groupby(["gvkey", "segment_srcdate"])["segment_sales"].transform("sum")
    grouped["revenue_share"] = grouped["segment_sales"] / grouped["snapshot_sales"]
    return grouped[grouped["revenue_share"].notna() & (grouped["revenue_share"] > 0)].copy()


def build_geo_vocab(tokens: pd.DataFrame, max_vocab: int) -> dict[str, int]:
    if max_vocab < 2:
        raise ValueError("max_vocab must be at least 2")
    weights = tokens.groupby("geo_label")["revenue_share"].sum().sort_values(ascending=False)
    labels = list(weights.head(max_vocab - 2).index)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab.update({label: index + 2 for index, label in enumerate(labels)})
    return vocab


def build_panel_frame(panel: pd.DataFrame, *, holdout_start: str = "2026-01-01") -> tuple[pd.DataFrame, list[str]]:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["segment_srcdate"] = pd.to_datetime(df["segment_srcdate"], errors="coerce")
    df["gvkey"] = df["gvkey"].astype(str)
    if TARGET not in df.columns:
        raise KeyError(f"Required target column missing: {TARGET}")
    df = df[df["date"].notna() & df["segment_srcdate"].notna() & (df["date"] < pd.Timestamp(holdout_start))].copy()

    needed = CONTROL_FEATURES + [TARGET, "mktcap", "at", "ceq", "sale", "ni", "capx", "xrd", "dltt", "dlc"]
    for col in set(needed):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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

    control_features = [feature for feature in CONTROL_FEATURES if feature in df.columns]
    keep = ["gvkey", "permno", "date", "segment_srcdate", TARGET] + control_features
    df = df[keep].replace([np.inf, -np.inf], np.nan)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype("float64")
    for feature in control_features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").astype("float64")
    df = df[df[TARGET].notna()].copy().reset_index(drop=True)
    return df, control_features


def encode_panel_sets(
    frame: pd.DataFrame,
    tokens: pd.DataFrame,
    vocab: dict[str, int],
    *,
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ranked = tokens.sort_values(["gvkey", "segment_srcdate", "revenue_share"], ascending=[True, True, False]).copy()
    ranked["segment_rank"] = ranked.groupby(["gvkey", "segment_srcdate"]).cumcount()
    ranked = ranked[ranked["segment_rank"] < max_segments].copy()
    ranked["geo_id"] = ranked["geo_label"].map(vocab).fillna(vocab["<UNK>"]).astype("int64")

    id_wide = ranked.pivot_table(index=["gvkey", "segment_srcdate"], columns="segment_rank", values="geo_id", aggfunc="first")
    share_wide = ranked.pivot_table(index=["gvkey", "segment_srcdate"], columns="segment_rank", values="revenue_share", aggfunc="first")
    expected_cols = list(range(max_segments))
    id_wide = id_wide.reindex(columns=expected_cols).fillna(0).astype("int64")
    share_wide = share_wide.reindex(columns=expected_cols).fillna(0.0).astype("float32")
    id_wide.columns = [f"geo_id_{i}" for i in expected_cols]
    share_wide.columns = [f"share_{i}" for i in expected_cols]

    encoded = frame[["gvkey", "segment_srcdate"]].merge(
        pd.concat([id_wide, share_wide], axis=1).reset_index(),
        on=["gvkey", "segment_srcdate"],
        how="left",
    )
    id_cols = [f"geo_id_{i}" for i in expected_cols]
    share_cols = [f"share_{i}" for i in expected_cols]
    geo_ids = encoded[id_cols].fillna(0).to_numpy(dtype=np.int64)
    shares = encoded[share_cols].fillna(0.0).to_numpy(dtype=np.float32)
    matched = (geo_ids != 0).any(axis=1)
    checks = {
        "encoded_rows": int(len(frame)),
        "matched_set_rows": int(matched.sum()),
        "matched_set_rate": float(matched.mean()) if len(matched) else None,
        "max_segments": int(max_segments),
        "vocab_size": int(len(vocab)),
        "token_snapshots": int(tokens[["gvkey", "segment_srcdate"]].drop_duplicates().shape[0]),
    }
    return geo_ids, shares, checks


def transform_controls(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[1] == 0:
        return train.astype("float32"), val.astype("float32")
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train_imp = np.where(np.isnan(train), med, train)
    val_imp = np.where(np.isnan(val), med, val)
    mean = train_imp.mean(axis=0)
    std = train_imp.std(axis=0)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    return ((train_imp - mean) / std).astype("float32"), ((val_imp - mean) / std).astype("float32")


def fit_predict_deepsets(
    frame: pd.DataFrame,
    geo_ids: np.ndarray,
    shares: np.ndarray,
    control_features: list[str],
    folds: list[dict[str, Any]],
    *,
    variants: list[str],
    vocab_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_train_rows_per_fold: int | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class DeepSetsRegressor(nn.Module):
        def __init__(self, num_geo_tokens: int, control_dim: int, embedding_dim: int = 16, hidden_dim: int = 64):
            super().__init__()
            self.embedding = nn.Embedding(num_geo_tokens, embedding_dim, padding_idx=0)
            self.phi = nn.Sequential(
                nn.Linear(embedding_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.rho = nn.Sequential(
                nn.Linear(hidden_dim + control_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, ids, revenue_shares, controls):
            emb = self.embedding(ids)
            token_x = torch.cat([emb, revenue_shares.unsqueeze(-1)], dim=-1)
            mask = ids.ne(0).unsqueeze(-1)
            token_repr = self.phi(token_x) * mask
            denom = mask.sum(dim=1).clamp_min(1)
            pooled = token_repr.sum(dim=1) / denom
            if controls.shape[1] > 0:
                pooled = torch.cat([pooled, controls], dim=1)
            return self.rho(pooled).squeeze(-1)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dates = pd.to_datetime(frame["date"])
    target = frame[TARGET].to_numpy(dtype=np.float32)
    controls_all = frame[control_features].to_numpy(dtype=np.float32) if control_features else np.zeros((len(frame), 0), dtype=np.float32)
    predictions_by_variant: dict[str, list[pd.DataFrame]] = {variant: [] for variant in variants}

    for variant in variants:
        variant_controls = [feature for feature in SET_VARIANTS[variant] if feature in control_features]
        control_idx = [control_features.index(feature) for feature in variant_controls]
        variant_controls_all = controls_all[:, control_idx] if control_idx else np.zeros((len(frame), 0), dtype=np.float32)
        for fold_index, fold in enumerate(folds):
            train_mask = (dates >= fold["train_start"]) & (dates <= fold["train_end"])
            val_mask = (dates >= fold["validation_start"]) & (dates <= fold["validation_end"])
            train_idx = np.flatnonzero(train_mask.to_numpy())
            val_idx = np.flatnonzero(val_mask.to_numpy())
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            if max_train_rows_per_fold and len(train_idx) > max_train_rows_per_fold:
                train_idx = rng.choice(train_idx, size=max_train_rows_per_fold, replace=False)
            y_train = target[train_idx]
            y_mean = float(y_train.mean())
            y_std = float(y_train.std())
            if not np.isfinite(y_std) or y_std <= 0:
                y_std = 1.0
            y_train_scaled = ((y_train - y_mean) / y_std).astype("float32")
            train_controls, val_controls = transform_controls(variant_controls_all[train_idx], variant_controls_all[val_idx])

            train_ds = TensorDataset(
                torch.as_tensor(geo_ids[train_idx], dtype=torch.long),
                torch.as_tensor(shares[train_idx], dtype=torch.float32),
                torch.as_tensor(train_controls, dtype=torch.float32),
                torch.as_tensor(y_train_scaled, dtype=torch.float32),
            )
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            model = DeepSetsRegressor(vocab_size, train_controls.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            loss_fn = nn.MSELoss()
            model.train()
            for _ in range(epochs):
                for batch_ids, batch_shares, batch_controls, batch_y in loader:
                    batch_ids = batch_ids.to(device, non_blocking=True)
                    batch_shares = batch_shares.to(device, non_blocking=True)
                    batch_controls = batch_controls.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    loss = loss_fn(model(batch_ids, batch_shares, batch_controls), batch_y)
                    loss.backward()
                    opt.step()

            model.eval()
            val_ds = TensorDataset(
                torch.as_tensor(geo_ids[val_idx], dtype=torch.long),
                torch.as_tensor(shares[val_idx], dtype=torch.float32),
                torch.as_tensor(val_controls, dtype=torch.float32),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
            pred_chunks: list[np.ndarray] = []
            with torch.no_grad():
                for batch_ids, batch_shares, batch_controls in val_loader:
                    raw = model(batch_ids.to(device), batch_shares.to(device), batch_controls.to(device)).detach().cpu().numpy()
                    pred_chunks.append(raw.astype("float32") * y_std + y_mean)
            pred = np.concatenate(pred_chunks)
            pred_frame = frame.iloc[val_idx][["gvkey", "permno", "date", TARGET]].copy()
            pred_frame["prediction"] = pred
            pred_frame["fold_year"] = fold["fold_year"]
            pred_frame["train_rows"] = int(len(train_idx))
            pred_frame["variant"] = variant
            predictions_by_variant[variant].append(pred_frame)

    return {
        variant: pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        for variant, chunks in predictions_by_variant.items()
    }


def render_figures(figures_dir: Path, variant: str, spread: pd.DataFrame, rank_ic: pd.DataFrame) -> dict[str, str | None]:
    plt = setup_matplotlib()
    label = variant.replace("_", " ").title()
    outputs: dict[str, str | None] = {"prediction_spread_figure": None, "rank_ic_figure": None}
    if len(spread):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.plot(spread["date"], spread["cum_q5_minus_q1"], color="#2A7F62", linewidth=2.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Deep Sets {label}: Predicted Q5 - Q1")
        ax.set_ylabel("Cumulative return")
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        path = figures_dir / f"deepsets_{variant}_prediction_spread_cumulative.png"
        fig.savefig(path)
        plt.close(fig)
        outputs["prediction_spread_figure"] = str(path)
    if len(rank_ic):
        ric = rank_ic.copy()
        ric["rolling_12m"] = ric["rank_ic"].rolling(12, min_periods=3).mean()
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(ric["date"], ric["rank_ic"], color="#8E6C8A", alpha=0.45, width=20)
        ax.plot(ric["date"], ric["rolling_12m"], color="#273B47", linewidth=2.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Deep Sets {label}: Monthly Rank IC")
        ax.set_ylabel("Rank IC")
        fig.autofmt_xdate(rotation=25)
        fig.tight_layout()
        path = figures_dir / f"deepsets_{variant}_rank_ic.png"
        fig.savefig(path)
        plt.close(fig)
        outputs["rank_ic_figure"] = str(path)
    return outputs


def summarize_variant(variant: str, predictions: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rank_ic = monthly_rank_ic(predictions) if len(predictions) else pd.DataFrame()
    quintiles, spread = prediction_quintile_returns(predictions) if len(predictions) else (pd.DataFrame(), pd.DataFrame())
    predictions_path = tables_dir / f"deepsets_{variant}_predictions.parquet"
    rank_ic_path = tables_dir / f"deepsets_{variant}_rank_ic.csv"
    spread_path = tables_dir / f"deepsets_{variant}_prediction_spread.csv"
    quintiles_path = tables_dir / f"deepsets_{variant}_prediction_quintile_returns.csv"
    atomic_write_parquet(predictions, predictions_path)
    rank_ic.to_csv(rank_ic_path, index=False)
    spread.to_csv(spread_path, index=False)
    quintiles.to_csv(quintiles_path, index=False)
    figure_outputs = render_figures(figures_dir, variant, spread, rank_ic)
    checks = {
        "prediction_rows": int(len(predictions)),
        "rank_ic_months": int(len(rank_ic)),
        "quintile_months": int(spread["date"].nunique()) if len(spread) else 0,
        "mean_rank_ic": float(rank_ic["rank_ic"].mean()) if len(rank_ic) else None,
        "t_rank_ic": t_stat(rank_ic["rank_ic"]) if len(rank_ic) else None,
        "mean_q5_minus_q1": float(spread["q5_minus_q1"].mean()) if len(spread) else None,
        "t_q5_minus_q1": t_stat(spread["q5_minus_q1"]) if len(spread) else None,
    }
    status = "ok" if len(predictions) and len(rank_ic) else "needs_review"
    outputs = {
        "predictions": str(predictions_path),
        "rank_ic": str(rank_ic_path),
        "spread": str(spread_path),
        "quintile_returns": str(quintiles_path),
        **figure_outputs,
    }
    return {"checks": checks, "outputs": outputs, "status": status}, {"variant": variant, "status": status, **checks}


def run_segment_set_model(
    project_root: Path,
    raw_run_id: str,
    panel_run_id: str,
    run_id: str,
    *,
    variants: list[str],
    max_segments: int,
    max_vocab: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    min_train_months: int,
    max_train_rows_per_fold: int | None,
    seed: int,
) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    raw_root = ensure_within(project_root, project_root / "data" / "raw" / raw_run_id)
    panel_path = ensure_within(project_root, project_root / "data" / "interim" / panel_run_id / "monthly_panel.parquet")
    tables_dir = ensure_within(project_root, project_root / "artifacts" / "tables" / run_id)
    figures_dir = ensure_within(project_root, project_root / "artifacts" / "figures_static" / run_id)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    frame, control_features = build_panel_frame(panel)
    segments = read_shards(raw_root, "segments")
    tokens = prepare_segment_tokens(segments)
    vocab = build_geo_vocab(tokens, max_vocab)
    geo_ids, shares, set_checks = encode_panel_sets(frame, tokens, vocab, max_segments=max_segments)
    matched = (geo_ids != 0).any(axis=1)
    frame = frame.loc[matched].reset_index(drop=True)
    geo_ids = geo_ids[matched]
    shares = shares[matched]
    folds = make_yearly_folds(frame["date"], min_train_months=min_train_months)
    predictions = fit_predict_deepsets(
        frame,
        geo_ids,
        shares,
        control_features,
        folds,
        variants=variants,
        vocab_size=len(vocab),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_train_rows_per_fold=max_train_rows_per_fold,
        seed=seed,
    )
    variant_outputs: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for variant, pred in predictions.items():
        variant_manifest, summary_row = summarize_variant(variant, pred, tables_dir, figures_dir)
        variant_outputs[variant] = variant_manifest
        summary_rows.append(summary_row)
    summary = pd.DataFrame(summary_rows)
    summary_path = tables_dir / "deepsets_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "run_id": run_id,
        "raw_run_id": raw_run_id,
        "panel_run_id": panel_run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "model": "deep_sets_segment_encoder",
        "variants": variants,
        "parameters": {
            "max_segments": int(max_segments),
            "max_vocab": int(max_vocab),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "min_train_months": int(min_train_months),
            "max_train_rows_per_fold": max_train_rows_per_fold,
            "seed": int(seed),
        },
        "inputs": {"panel": str(panel_path), "panel_rows": int(len(panel)), "frame_rows": int(len(frame)), "segment_rows": int(len(segments))},
        "set_encoding": set_checks,
        "control_features": control_features,
        "validation": {
            "scheme": "expanding_train_yearly_validation",
            "folds": [
                {key: (str(value.date()) if hasattr(value, "date") else value) for key, value in fold.items()}
                for fold in folds
            ],
        },
        "checks": {
            "variants_ok": int((summary["status"] == "ok").sum()) if len(summary) else 0,
            "variants_total": int(len(summary)),
            "best_rank_ic_variant": summary.sort_values("mean_rank_ic", ascending=False)["variant"].iloc[0] if len(summary) else None,
            "best_spread_variant": summary.sort_values("mean_q5_minus_q1", ascending=False)["variant"].iloc[0] if len(summary) else None,
        },
        "outputs": {"summary": str(summary_path), "variants": variant_outputs},
        "status": "ok" if len(summary) and (summary["status"] == "ok").all() else "needs_review",
    }
    atomic_write_json(paths.manifests / "segment_set_model.json", manifest)
    atomic_write_text(paths.reports / "segment_set_model_report.md", report_text(manifest, summary))
    return manifest


def report_text(manifest: dict[str, Any], summary: pd.DataFrame) -> str:
    lines = [
        "# Segment Set Model Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Panel run ID: `{manifest['panel_run_id']}`",
        f"- Status: `{manifest['status']}`",
        f"- Model: `{manifest['model']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Rank IC | Rank IC t | Q5-Q1 | Q5-Q1 t |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| `{row['variant']}` | {row['mean_rank_ic']} | {row['t_rank_ic']} | "
            f"{row['mean_q5_minus_q1']} | {row['t_q5_minus_q1']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--raw-run-id", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variants", default="set_only,set_plus_controls")
    parser.add_argument("--max-segments", type=int, default=12)
    parser.add_argument("--max-vocab", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-train-months", type=int, default=36)
    parser.add_argument("--max-train-rows-per-fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_segment_set_model(
        project_root,
        args.raw_run_id,
        args.panel_run_id,
        args.run_id,
        variants=parse_variants(args.variants),
        max_segments=args.max_segments,
        max_vocab=args.max_vocab,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_train_months=args.min_train_months,
        max_train_rows_per_fold=args.max_train_rows_per_fold,
        seed=args.seed,
    )
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'segment_set_model.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
