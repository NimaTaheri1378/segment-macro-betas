from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from segment_macro_betas.io_utils import atomic_write_json, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


DISPLAY_MODEL_COLUMNS = {
    "model_family": "Model",
    "variant": "Variant",
    "architecture": "Architecture",
    "prediction_rows": "Prediction rows",
    "mean_rank_ic": "Rank IC",
    "t_rank_ic": "t(Rank IC)",
    "mean_q5_minus_q1": "Q5-Q1",
    "t_q5_minus_q1": "t(Q5-Q1)",
    "review_note": "Review note",
}

DISPLAY_FACTOR_COLUMNS = {
    "model_family": "Model",
    "variant": "Variant",
    "months": "Months",
    "mean_gross_q5_minus_q1": "Gross Q5-Q1",
    "t_gross_q5_minus_q1": "t(Gross)",
    "mean_net_q5_minus_q1": "Net Q5-Q1",
    "t_net_q5_minus_q1": "t(Net)",
    "mean_turnover": "Turnover",
    "gross_alpha": "Gross alpha",
    "gross_alpha_t": "t(Gross alpha)",
    "net_alpha": "Net alpha",
    "net_alpha_t": "t(Net alpha)",
    "alpha_months": "Alpha months",
    "review_note": "Review note",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table missing: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def as_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_value(value: Any, *, digits: int = 4) -> str:
    number = as_number(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return ""
    return f"{int(round(number)):,}"


def model_review_note(row: pd.Series) -> str:
    rank_t = as_number(row.get("t_rank_ic"))
    spread_t = as_number(row.get("t_q5_minus_q1"))
    rank_ok = rank_t is not None and rank_t > 2.0
    spread_ok = spread_t is not None and spread_t > 2.0
    if rank_ok and spread_ok:
        return "positive rank and spread diagnostic"
    if rank_ok:
        return "positive rank diagnostic"
    if spread_ok:
        return "positive spread diagnostic"
    return "weak diagnostic"


def factor_review_note(row: pd.Series) -> str:
    alpha_t = as_number(row.get("gross_alpha_t"))
    net_t = as_number(row.get("t_net_q5_minus_q1"))
    alpha_ok = alpha_t is not None and alpha_t > 2.0
    net_ok = net_t is not None and net_t > 2.0
    if alpha_ok and net_ok:
        return "positive alpha and net-spread diagnostic"
    if alpha_ok:
        return "positive alpha diagnostic"
    if net_ok:
        return "positive net-spread diagnostic"
    return "weak diagnostic"


def normalize_model_family(value: Any) -> str:
    text = str(value).strip()
    if text.lower() == "lgbm":
        return "LightGBM"
    if text.lower() == "deepsets":
        return "Deep Sets"
    return text


def build_model_comparison(lgbm_summary: pd.DataFrame, set_summary: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not lgbm_summary.empty:
        lgbm = lgbm_summary.copy()
        lgbm["model_family"] = "LightGBM"
        lgbm["architecture"] = "gradient_boosted_trees"
        frames.append(lgbm)
    if not set_summary.empty:
        set_models = set_summary.copy()
        set_models["model_family"] = "Deep Sets"
        if "architecture" not in set_models.columns:
            set_models["architecture"] = "deep_sets"
        frames.append(set_models)
    if not frames:
        return pd.DataFrame(columns=list(DISPLAY_MODEL_COLUMNS))

    out = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["prediction_rows", "mean_rank_ic", "t_rank_ic", "mean_q5_minus_q1", "t_q5_minus_q1"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["review_note"] = out.apply(model_review_note, axis=1)
    keep = [col for col in DISPLAY_MODEL_COLUMNS if col in out.columns]
    return out[keep].sort_values(["model_family", "variant"]).reset_index(drop=True)


def build_factor_alpha_table(factor_summary: pd.DataFrame) -> pd.DataFrame:
    if factor_summary.empty:
        return pd.DataFrame(columns=list(DISPLAY_FACTOR_COLUMNS))
    out = factor_summary.copy()
    if "model_family" in out.columns:
        out["model_family"] = out["model_family"].map(normalize_model_family)
    for col in DISPLAY_FACTOR_COLUMNS:
        if col not in {"model_family", "variant", "review_note"} and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["review_note"] = out.apply(factor_review_note, axis=1)
    keep = [col for col in DISPLAY_FACTOR_COLUMNS if col in out.columns]
    return out[keep].sort_values(["model_family", "variant"]).reset_index(drop=True)


def table_notes(*, panel_run_id: str, cost_bps: float, nw_lag: int) -> list[str]:
    return [
        "All entries are private diagnostics for the 2006-2025 development sample; the 2026 holdout remains untouched.",
        f"Panel provenance: {panel_run_id}; segment disclosures are activated using filing-date timing where available.",
        f"Transaction-cost stress subtracts {cost_bps:g} bps per one-way long-short turnover and is not a capacity model.",
        f"Factor-alpha t-statistics use Newey-West lag {nw_lag} with available benchmark factors.",
        "Macro-beta and vintage-safe macro-interaction claims remain blocked until official macro API execution and tensor review are complete.",
    ]


def display_frame(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    available = [col for col in columns if col in frame.columns]
    display = frame[available].rename(columns=columns).copy()
    for col in display.columns:
        raw_col = next((key for key, label in columns.items() if label == col), col)
        if raw_col in {"prediction_rows", "months", "alpha_months"}:
            display[col] = display[col].map(fmt_int)
        elif raw_col not in {"model_family", "variant", "architecture", "review_note"}:
            display[col] = display[col].map(fmt_value)
        else:
            display[col] = display[col].fillna("").astype(str)
    return display


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = [str(col) for col in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def latex_table(frame: pd.DataFrame, *, caption: str, label: str) -> str:
    if frame.empty:
        return "% No rows.\n"
    columns = "l" * len(frame.columns)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(map(str, frame.columns)) + " \\\\",
        "\\midrule",
    ]
    for row in frame.to_numpy():
        cells = [str(value).replace("_", "\\_").replace("%", "\\%") for value in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render_report(title: str, display: pd.DataFrame, notes: list[str]) -> str:
    lines = [f"# {title}", "", markdown_table(display), "", "## Notes", ""]
    lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return "\n".join(lines)


def validate_publication_tables(
    model_table: pd.DataFrame,
    factor_table: pd.DataFrame,
    claim_validation: pd.DataFrame,
    notes: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    rows.append(
        {
            "check": "model_table_nonempty",
            "status": "pass" if len(model_table) else "fail",
            "detail": f"rows={len(model_table)}",
        }
    )
    rows.append(
        {
            "check": "factor_table_nonempty",
            "status": "pass" if len(factor_table) else "fail",
            "detail": f"rows={len(factor_table)}",
        }
    )
    duplicate_model_keys = int(model_table.duplicated(["model_family", "variant"]).sum()) if len(model_table) else 0
    duplicate_factor_keys = int(factor_table.duplicated(["model_family", "variant"]).sum()) if len(factor_table) else 0
    rows.append(
        {
            "check": "unique_model_rows",
            "status": "pass" if duplicate_model_keys == 0 else "fail",
            "detail": f"duplicates={duplicate_model_keys}",
        }
    )
    rows.append(
        {
            "check": "unique_factor_rows",
            "status": "pass" if duplicate_factor_keys == 0 else "fail",
            "detail": f"duplicates={duplicate_factor_keys}",
        }
    )
    fail_count = int((claim_validation.get("status", pd.Series(dtype=str)) == "fail").sum()) if len(claim_validation) else 0
    rows.append(
        {
            "check": "claim_validation_no_failures",
            "status": "pass" if fail_count == 0 else "fail",
            "detail": f"failures={fail_count}",
        }
    )
    note_text = " ".join(notes).lower()
    for token in ["diagnostic", "2026", "macro", "turnover"]:
        rows.append(
            {
                "check": f"notes_include_{token}",
                "status": "pass" if token in note_text else "fail",
                "detail": token,
            }
        )
    return pd.DataFrame(rows)


def run_publication_tables(
    project_root: Path,
    run_id: str,
    *,
    panel_run_id: str,
    lgbm_run_id: str,
    set_run_id: str,
    factor_run_id: str,
    claim_run_id: str,
    cost_bps: float,
    nw_lag: int,
) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    tables_root = ensure_within(project_root, project_root / "artifacts" / "tables")
    out_dir = ensure_within(project_root, tables_root / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    lgbm_summary = read_required_csv(tables_root / lgbm_run_id / "lgbm_summary.csv")
    set_summary = read_required_csv(tables_root / set_run_id / "deepsets_summary.csv")
    factor_summary = read_required_csv(tables_root / factor_run_id / "factor_robustness_summary.csv")
    claim_validation = read_optional_csv(tables_root / claim_run_id / "claim_validation.csv")

    model_table = build_model_comparison(lgbm_summary, set_summary)
    factor_table = build_factor_alpha_table(factor_summary)
    notes = table_notes(panel_run_id=panel_run_id, cost_bps=cost_bps, nw_lag=nw_lag)
    checks = validate_publication_tables(model_table, factor_table, claim_validation, notes)

    model_path = out_dir / "publication_model_comparison.csv"
    factor_path = out_dir / "publication_factor_alpha.csv"
    checks_path = out_dir / "publication_review_checks.csv"
    model_table.to_csv(model_path, index=False)
    factor_table.to_csv(factor_path, index=False)
    checks.to_csv(checks_path, index=False)

    model_display = display_frame(model_table, DISPLAY_MODEL_COLUMNS)
    factor_display = display_frame(factor_table, DISPLAY_FACTOR_COLUMNS)
    model_report_path = paths.reports / "publication_model_comparison.md"
    factor_report_path = paths.reports / "publication_factor_alpha.md"
    model_tex_path = paths.reports / "publication_model_comparison.tex"
    factor_tex_path = paths.reports / "publication_factor_alpha.tex"
    atomic_write_text(model_report_path, render_report("Publication-Style Model Comparison Diagnostics", model_display, notes))
    atomic_write_text(factor_report_path, render_report("Publication-Style Factor Alpha and Cost Diagnostics", factor_display, notes))
    atomic_write_text(
        model_tex_path,
        latex_table(model_display, caption="Model comparison diagnostics", label="tab:model_comparison_diagnostics"),
    )
    atomic_write_text(
        factor_tex_path,
        latex_table(factor_display, caption="Factor alpha and cost diagnostics", label="tab:factor_alpha_cost_diagnostics"),
    )

    fail_count = int((checks["status"] == "fail").sum())
    manifest = {
        "run_id": run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "inputs": {
            "panel_run_id": panel_run_id,
            "lgbm_run_id": lgbm_run_id,
            "set_run_id": set_run_id,
            "factor_run_id": factor_run_id,
            "claim_run_id": claim_run_id,
        },
        "parameters": {"cost_bps": float(cost_bps), "nw_lag": int(nw_lag)},
        "outputs": {
            "model_table": str(model_path),
            "factor_table": str(factor_path),
            "review_checks": str(checks_path),
            "model_report": str(model_report_path),
            "factor_report": str(factor_report_path),
            "model_latex": str(model_tex_path),
            "factor_latex": str(factor_tex_path),
        },
        "checks": {
            "model_rows": int(len(model_table)),
            "factor_rows": int(len(factor_table)),
            "review_failures": fail_count,
            "claim_validation_rows": int(len(claim_validation)),
            "claim_validation_failures": int((claim_validation.get("status", pd.Series(dtype=str)) == "fail").sum()) if len(claim_validation) else 0,
        },
        "status": "ok" if fail_count == 0 else "needs_review",
    }
    atomic_write_json(paths.manifests / "publication_tables.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--lgbm-run-id", required=True)
    parser.add_argument("--set-run-id", required=True)
    parser.add_argument("--factor-run-id", required=True)
    parser.add_argument("--claim-run-id", required=True)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--nw-lag", type=int, default=6)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_publication_tables(
        project_root,
        args.run_id,
        panel_run_id=args.panel_run_id,
        lgbm_run_id=args.lgbm_run_id,
        set_run_id=args.set_run_id,
        factor_run_id=args.factor_run_id,
        claim_run_id=args.claim_run_id,
        cost_bps=args.cost_bps,
        nw_lag=args.nw_lag,
    )
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'publication_tables.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
