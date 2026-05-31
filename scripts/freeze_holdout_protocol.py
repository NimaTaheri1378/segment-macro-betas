from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PANEL_RUN_ID = "20260531T003936Z_panel_filing"
DEFAULT_LGBM_RUN_ID = "20260531T_lgbm_macro_nonfred_v3"
DEFAULT_FACTOR_RUN_ID = "20260531T_factor_robustness_macro_nonfred"
DEFAULT_CLAIM_RUN_ID = "20260531T_claim_ledger_macro_nonfred"
DEFAULT_PUBLICATION_RUN_ID = "20260531T_publication_tables_macro_nonfred"
DEFAULT_VISUAL_RUN_ID = "20260531T_visual_pack_macro_nonfred_v2"
DEFAULT_MODEL_FAMILY = "lgbm"
DEFAULT_VARIANT = "all_plus_macro"
HOLDOUT_START = "2026-01-01"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def add(rows: list[dict[str, str]], check: str, status: str, detail: str) -> None:
    rows.append({"check": check, "status": status, "detail": detail})


def require(rows: list[dict[str, str]], check: str, condition: bool, detail: str) -> None:
    add(rows, check, "pass" if condition else "fail", detail)


def metric(row: pd.Series, name: str) -> float | None:
    value = pd.to_numeric(row.get(name), errors="coerce")
    return None if pd.isna(value) else float(value)


def freeze_protocol(
    project_root: Path,
    *,
    panel_run_id: str,
    lgbm_run_id: str,
    factor_run_id: str,
    claim_run_id: str,
    publication_run_id: str,
    visual_run_id: str,
    model_family: str,
    variant: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    panel_manifest = read_json(root / "runs" / panel_run_id / "manifests" / "monthly_panel.json")
    lgbm_manifest = read_json(root / "runs" / lgbm_run_id / "manifests" / "lgbm_benchmark.json")
    factor_manifest = read_json(root / "runs" / factor_run_id / "manifests" / "factor_robustness.json")
    claim_manifest = read_json(root / "runs" / claim_run_id / "manifests" / "claim_ledger.json")
    publication_manifest = read_json(root / "runs" / publication_run_id / "manifests" / "publication_tables.json")
    visual_manifest = read_json(root / "runs" / visual_run_id / "manifests" / "visual_pack.json")

    lgbm_summary_path = root / "artifacts" / "tables" / lgbm_run_id / "lgbm_summary.csv"
    factor_summary_path = root / "artifacts" / "tables" / factor_run_id / "factor_robustness_summary.csv"
    lgbm_summary = pd.read_csv(lgbm_summary_path)
    factor_summary = pd.read_csv(factor_summary_path)

    rows: list[dict[str, str]] = []
    require(rows, "panel_status_ok", panel_manifest.get("status") == "ok", f"status={panel_manifest.get('status')}")
    require(rows, "holdout_unopened_by_panel_max_date", nested(panel_manifest, "checks", "max_date") == "2025-12-31", f"max_date={nested(panel_manifest, 'checks', 'max_date')}")
    require(rows, "activation_no_violations", nested(panel_manifest, "checks", "activation_rule_violations") == 0, f"violations={nested(panel_manifest, 'checks', 'activation_rule_violations')}")
    require(rows, "lgbm_status_ok", lgbm_manifest.get("status") == "ok", f"status={lgbm_manifest.get('status')}")
    variant_manifest = nested(lgbm_manifest, "outputs", "variants", variant, default={})
    require(rows, "selected_variant_exists", bool(variant_manifest), f"variant={variant}")
    require(rows, "selected_variant_ok", variant_manifest.get("status") == "ok", f"status={variant_manifest.get('status')}")
    require(rows, "selected_by_development_spread", nested(lgbm_manifest, "checks", "best_spread_variant") == variant, f"best_spread_variant={nested(lgbm_manifest, 'checks', 'best_spread_variant')}")
    require(rows, "factor_status_ok", factor_manifest.get("status") == "ok", f"status={factor_manifest.get('status')}")
    require(rows, "claim_validation_clean", nested(claim_manifest, "checks", "validation_failures") == 0, f"checks={claim_manifest.get('checks')}")
    require(rows, "publication_review_clean", nested(publication_manifest, "checks", "review_failures") == 0, f"checks={publication_manifest.get('checks')}")
    require(rows, "visual_pack_ok", visual_manifest.get("status") == "ok", f"status={visual_manifest.get('status')}")

    lgbm_row = lgbm_summary.loc[lgbm_summary["variant"] == variant]
    factor_row = factor_summary.loc[(factor_summary["model_family"] == model_family) & (factor_summary["variant"] == variant)]
    require(rows, "selected_variant_in_lgbm_summary", len(lgbm_row) == 1, f"rows={len(lgbm_row)}")
    require(rows, "selected_variant_in_factor_summary", len(factor_row) == 1, f"rows={len(factor_row)}")

    lgbm_metrics: dict[str, Any] = {}
    factor_metrics: dict[str, Any] = {}
    if len(lgbm_row) == 1:
        row = lgbm_row.iloc[0]
        lgbm_metrics = {
            "mean_rank_ic": metric(row, "mean_rank_ic"),
            "t_rank_ic": metric(row, "t_rank_ic"),
            "mean_q5_minus_q1": metric(row, "mean_q5_minus_q1"),
            "t_q5_minus_q1": metric(row, "t_q5_minus_q1"),
            "prediction_rows": int(row["prediction_rows"]),
        }
    if len(factor_row) == 1:
        row = factor_row.iloc[0]
        factor_metrics = {
            "mean_net_q5_minus_q1": metric(row, "mean_net_q5_minus_q1"),
            "t_net_q5_minus_q1": metric(row, "t_net_q5_minus_q1"),
            "gross_alpha": metric(row, "gross_alpha"),
            "gross_alpha_t": metric(row, "gross_alpha_t"),
            "alpha_months": int(row["alpha_months"]),
        }

    failures = [row for row in rows if row["status"] == "fail"]
    status = "frozen" if not failures else "needs_review"
    return {
        "created_utc": now_iso(),
        "status": status,
        "holdout_start": HOLDOUT_START,
        "holdout_opened": False,
        "selected_model": {
            "family": model_family,
            "run_id": lgbm_run_id,
            "variant": variant,
            "selection_rule": "highest development-sample mean Q5-Q1 among current macro-aware LightGBM variants",
        },
        "inputs": {
            "panel_run_id": panel_run_id,
            "lgbm_run_id": lgbm_run_id,
            "factor_run_id": factor_run_id,
            "claim_run_id": claim_run_id,
            "publication_run_id": publication_run_id,
            "visual_run_id": visual_run_id,
        },
        "development_metrics": {
            "lgbm": lgbm_metrics,
            "factor_robustness": factor_metrics,
        },
        "checks": {
            "passed": sum(row["status"] == "pass" for row in rows),
            "failed": len(failures),
            "total": len(rows),
        },
        "rows": rows,
        "blocked_claims": [
            "Do not report 2026 performance until an explicit future holdout run is authorized.",
            "Do not promote revision-safe macro claims until a true realtime/vintage macro run is executed.",
        ],
    }


def write_outputs(project_root: Path, run_id: str, manifest: dict[str, Any]) -> None:
    run_root = project_root / "runs" / run_id
    manifest_dir = run_root / "manifests"
    report_dir = run_root / "reports"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "holdout_protocol.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Holdout Protocol Freeze",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Holdout opened: `{manifest['holdout_opened']}`",
        f"- Holdout start: `{manifest['holdout_start']}`",
        f"- Selected model: `{manifest['selected_model']['family']}:{manifest['selected_model']['run_id']}:{manifest['selected_model']['variant']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in manifest["rows"]:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| `{row['check']}` | `{row['status']}` | {detail} |")
    lines.extend(["", "## Blocked Claims", ""])
    for item in manifest["blocked_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    (report_dir / "holdout_protocol.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--panel-run-id", default=DEFAULT_PANEL_RUN_ID)
    parser.add_argument("--lgbm-run-id", default=DEFAULT_LGBM_RUN_ID)
    parser.add_argument("--factor-run-id", default=DEFAULT_FACTOR_RUN_ID)
    parser.add_argument("--claim-run-id", default=DEFAULT_CLAIM_RUN_ID)
    parser.add_argument("--publication-run-id", default=DEFAULT_PUBLICATION_RUN_ID)
    parser.add_argument("--visual-run-id", default=DEFAULT_VISUAL_RUN_ID)
    parser.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    manifest = freeze_protocol(
        root,
        panel_run_id=args.panel_run_id,
        lgbm_run_id=args.lgbm_run_id,
        factor_run_id=args.factor_run_id,
        claim_run_id=args.claim_run_id,
        publication_run_id=args.publication_run_id,
        visual_run_id=args.visual_run_id,
        model_family=args.model_family,
        variant=args.variant,
    )
    write_outputs(root, args.run_id, manifest)
    print(f"holdout_protocol_{manifest['status']}")
    print(json.dumps(manifest["checks"], sort_keys=True))
    return 0 if manifest["status"] == "frozen" else 1


if __name__ == "__main__":
    raise SystemExit(main())
