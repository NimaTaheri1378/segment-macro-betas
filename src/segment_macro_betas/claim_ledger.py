from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from segment_macro_betas.io_utils import atomic_write_json, atomic_write_text
from segment_macro_betas.paths import SCRATCH_PROJECT_ROOT, ensure_within, make_run_paths, require_project_root, resolve_project_root


FORBIDDEN_STRONG_WORDS = ["causes", "proves", "fully explains", "tradable", "arbitrage", "orthogonal"]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_run_ids(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        run_ids = raw
    else:
        run_ids = [piece.strip() for piece in raw.split(",")]
    out = [run_id for run_id in run_ids if run_id]
    if not out:
        raise ValueError("At least one run id is required.")
    return out


def read_set_summaries(tables_root: Path, set_run_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_id in set_run_ids:
        frame = read_optional_csv(tables_root / run_id / "deepsets_summary.csv")
        if frame.empty:
            continue
        frame = frame.copy()
        frame["model_run_id"] = run_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def best_row(frame: pd.DataFrame, column: str) -> dict[str, Any] | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() == 0:
        return None
    return frame.loc[values.idxmax()].to_dict()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def claim_row(
    module: str,
    claim: str,
    evidence_strength: str,
    source_artifacts: str,
    allowed_wording: str,
    forbidden_wording: str,
    caveats_next_gates: str,
) -> dict[str, str]:
    return {
        "module": module,
        "claim": claim,
        "evidence_strength": evidence_strength,
        "source_artifacts": source_artifacts,
        "allowed_wording": allowed_wording,
        "forbidden_wording": forbidden_wording,
        "caveats_next_gates": caveats_next_gates,
    }


def macro_execution_claim(lgbm_run_id: str) -> dict[str, str]:
    if "fred_initial_release" in lgbm_run_id:
        return claim_row(
            "Macro execution",
            "Limited FRED initial-release macro interactions are live for the included FRED series.",
            "passed diagnostic",
            f"{lgbm_run_id}:lgbm_summary.csv; paired macro_engine and macro_tensor manifests",
            (
                "The FRED initial-release macro chain uses realtime availability dates and can be described as "
                "revision-safe for the included FRED series."
            ),
            "The full FRED/BLS/BEA/EIA macro catalog is complete, or all macro interactions are final.",
            "Keep this claim limited to the included FRED series; broader official-source and 2026 holdout claims remain gated.",
        )
    if "macro_nonfred" in lgbm_run_id:
        return claim_row(
            "Macro execution",
            "Official non-FRED macro pulls are live with no-lookahead timing but not true vintage safety.",
            "blocked by missing true revision vintages",
            "configs/macro_series.yml; macro_engine and macro_tensor manifests",
            "BLS, BEA, and EIA official macro data have been pulled into a no-lookahead macro tensor.",
            "The non-FRED macro interactions are final revision-safe vintage interactions.",
            "Use true realtime or vintage macro sources before making final revision-safe macro-beta claims.",
        )
    return claim_row(
        "Macro execution",
        "Official macro API and true vintage macro-tensor claims remain gated.",
        "blocked by missing private secrets",
        "configs/macro_series.yml; macro dry-run manifests",
        "The public code can execute FRED/BLS/BEA/EIA pulls once untracked compute-host credentials are present.",
        "The current empirical results include final vintage-safe macro interactions.",
        "Run the live macro engine from an untracked `.env`, then build and review the macro tensor before making macro-beta claims.",
    )


def build_claim_ledger(
    *,
    panel_run_id: str,
    lgbm_run_id: str,
    set_run_id: str,
    factor_run_id: str,
    lgbm_summary: pd.DataFrame,
    set_summary: pd.DataFrame,
    factor_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    segment = lgbm_summary[lgbm_summary.get("variant", pd.Series(dtype=str)) == "segment_only"]
    if len(segment):
        row = segment.iloc[0]
        rows.append(
            claim_row(
                "LightGBM ablations",
                "Segment-only disclosure features contain cross-sectional ranking signal.",
                "passed diagnostic" if float(row.get("t_rank_ic", 0.0)) > 2 else "weak diagnostic",
                f"{lgbm_run_id}:lgbm_summary.csv; {panel_run_id}:monthly_panel",
                (
                    "In the filing-date panel, the segment-only LightGBM variant has a positive monthly rank IC "
                    f"of {fmt(row.get('mean_rank_ic'))} with t-stat {fmt(row.get('t_rank_ic'))}."
                ),
                "Segment disclosures alone dominate controls, prove underreaction, or establish tradable profits.",
                "Compare against macro-vintage tensors and untouched 2026 holdout before making final paper claims.",
            )
        )

    best_spread = best_row(lgbm_summary, "mean_q5_minus_q1")
    if best_spread:
        rows.append(
            claim_row(
                "LightGBM ablations",
                "Control-rich variants currently have stronger long-short spread diagnostics than segment-only features.",
                "passed diagnostic",
                f"{lgbm_run_id}:lgbm_summary.csv",
                (
                    f"The strongest gross Q5-Q1 spread among LightGBM variants is `{best_spread['variant']}` "
                    f"with mean spread {fmt(best_spread.get('mean_q5_minus_q1'))}."
                ),
                "The segment-only result is the dominant economic portfolio result.",
                "This is a diagnostic comparison of equal-weight prediction sorts, not an implementation-ready strategy.",
            )
        )

    if len(set_summary):
        set_only = set_summary[set_summary.get("variant", pd.Series(dtype=str)) == "set_only"]
        if len(set_only):
            row = set_only.iloc[0]
            strength = "passed diagnostic" if float(row.get("t_rank_ic", 0.0)) > 2 else "weak diagnostic"
            rows.append(
                claim_row(
                    "Segment-set models",
                    "The simple Deep Sets encoder validates the set-structured code path but does not dominate tabular models.",
                    strength,
                    f"{set_run_id}:deepsets_summary.csv",
                    (
                        "The set-only Deep Sets variant has mean rank IC "
                        f"{fmt(row.get('mean_rank_ic'))} and mean Q5-Q1 {fmt(row.get('mean_q5_minus_q1'))}."
                    ),
                    "Set models outperform all tabular benchmarks or establish a superior architecture.",
                    "Treat as first architecture evidence; full Set Transformer scaling remains optional and diagnostic.",
                )
            )
        transformer = set_summary[set_summary.get("variant", pd.Series(dtype=str)) == "set_transformer"]
        if len(transformer):
            row = transformer.iloc[0]
            strength = "passed diagnostic" if float(row.get("t_rank_ic", 0.0)) > 2 else "weak diagnostic"
            rows.append(
                claim_row(
                    "Segment-set models",
                    "The full Set Transformer run is executable on CUDA but does not dominate the tabular benchmarks.",
                    strength,
                    f"{row.get('model_run_id', set_run_id)}:deepsets_summary.csv",
                    (
                        "The full Set Transformer variant completed on CUDA with mean rank IC "
                        f"{fmt(row.get('mean_rank_ic'))} and mean Q5-Q1 {fmt(row.get('mean_q5_minus_q1'))}."
                    ),
                    "Set Transformer attention materially improves the economic long-short result.",
                    "Treat as a full-scale architecture diagnostic; it remains weaker than the current LightGBM spread diagnostics.",
                )
            )

    best_alpha = best_row(factor_summary, "gross_alpha_t")
    if best_alpha:
        rows.append(
            claim_row(
                "Factor robustness",
                "Some LightGBM variants retain positive factor-alpha diagnostics after benchmark-factor adjustment.",
                "passed diagnostic" if float(best_alpha.get("gross_alpha_t", 0.0)) > 2 else "weak diagnostic",
                f"{factor_run_id}:factor_robustness_summary.csv",
                (
                    f"The strongest gross alpha diagnostic is `{best_alpha['model_family']}:{best_alpha['variant']}` "
                    f"with monthly alpha {fmt(best_alpha.get('gross_alpha'))} and t-stat {fmt(best_alpha.get('gross_alpha_t'))}."
                ),
                "The strategy is orthogonal to factors, implementable after costs, or arbitrage-like.",
                "Alpha months are limited by factor availability; table notes must state costs, factors, and Newey-West lag.",
            )
        )

    best_net = best_row(factor_summary, "mean_net_q5_minus_q1")
    if best_net:
        rows.append(
            claim_row(
                "Turnover robustness",
                "The cost stress preserves positive net spread diagnostics for selected LightGBM variants.",
                "passed diagnostic" if float(best_net.get("t_net_q5_minus_q1", 0.0)) > 2 else "weak diagnostic",
                f"{factor_run_id}:factor_robustness_summary.csv",
                (
                    f"The strongest net Q5-Q1 diagnostic is `{best_net['model_family']}:{best_net['variant']}` "
                    f"with mean net spread {fmt(best_net.get('mean_net_q5_minus_q1'))} and t-stat {fmt(best_net.get('t_net_q5_minus_q1'))}."
                ),
                "The portfolio is tradable at institutional scale or capacity-safe.",
                "Direct turnover-cost stress is not a full transaction-cost/capacity model.",
            )
        )

    rows.append(macro_execution_claim(lgbm_run_id))
    return pd.DataFrame(rows)


def validate_claims(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in ledger.iterrows():
        allowed = str(row["allowed_wording"]).lower()
        forbidden_hits = [word for word in FORBIDDEN_STRONG_WORDS if word in allowed]
        strength = str(row["evidence_strength"])
        status = "blocked" if "blocked" in strength else "pass"
        if forbidden_hits:
            status = "fail"
        rows.append(
            {
                "row": int(idx),
                "module": row["module"],
                "status": status,
                "forbidden_hits": ",".join(forbidden_hits),
                "evidence_strength": strength,
            }
        )
    return pd.DataFrame(rows)


def build_table_inventory(run_ids: dict[str, str], tables_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family, run_id in run_ids.items():
        run_dir = tables_root / run_id
        files = sorted(path for path in run_dir.glob("*.csv")) if run_dir.exists() else []
        for path in files:
            try:
                frame = pd.read_csv(path, nrows=5)
                columns = ",".join(frame.columns.astype(str).tolist())
            except Exception:
                columns = ""
            rows.append({"family": family, "run_id": run_id, "table": path.name, "sample_columns": columns})
    return pd.DataFrame(rows)


def report_text(manifest: dict[str, Any], ledger: pd.DataFrame, validation: pd.DataFrame) -> str:
    lines = [
        "# Claim Ledger Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Status: `{manifest['status']}`",
        f"- Claim rows: `{len(ledger)}`",
        f"- Validation failures: `{int((validation['status'] == 'fail').sum())}`",
        f"- Blocked claims: `{int((validation['status'] == 'blocked').sum())}`",
        "",
        "## Allowed Claim Wording",
        "",
    ]
    for row in ledger.to_dict(orient="records"):
        lines.append(f"- **{row['module']}**: {row['allowed_wording']}")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "Private diagnostics are research status notes until final table review and the untouched 2026 holdout protocol are complete.",
            "",
        ]
    )
    return "\n".join(lines)


def run_claim_ledger(
    project_root: Path,
    run_id: str,
    panel_run_id: str,
    lgbm_run_id: str,
    set_run_ids: list[str],
    factor_run_id: str,
) -> dict[str, Any]:
    paths = make_run_paths(project_root, run_id)
    tables_root = ensure_within(project_root, project_root / "artifacts" / "tables")
    out_dir = ensure_within(project_root, tables_root / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    lgbm_summary = read_optional_csv(tables_root / lgbm_run_id / "lgbm_summary.csv")
    set_summary = read_set_summaries(tables_root, set_run_ids)
    factor_summary = read_optional_csv(tables_root / factor_run_id / "factor_robustness_summary.csv")
    ledger = build_claim_ledger(
        panel_run_id=panel_run_id,
        lgbm_run_id=lgbm_run_id,
        set_run_id=",".join(set_run_ids),
        factor_run_id=factor_run_id,
        lgbm_summary=lgbm_summary,
        set_summary=set_summary,
        factor_summary=factor_summary,
    )
    validation = validate_claims(ledger)
    inventory = build_table_inventory(
        {"lgbm": lgbm_run_id, **{f"set_{idx + 1}": value for idx, value in enumerate(set_run_ids)}, "factor": factor_run_id},
        tables_root,
    )

    ledger_path = out_dir / "claim_ledger.csv"
    validation_path = out_dir / "claim_validation.csv"
    inventory_path = out_dir / "table_inventory.csv"
    ledger.to_csv(ledger_path, index=False)
    validation.to_csv(validation_path, index=False)
    inventory.to_csv(inventory_path, index=False)

    status = "ok" if len(ledger) and not (validation["status"] == "fail").any() else "needs_review"
    manifest = {
        "run_id": run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "inputs": {
            "panel_run_id": panel_run_id,
            "lgbm_run_id": lgbm_run_id,
            "set_run_ids": set_run_ids,
            "factor_run_id": factor_run_id,
        },
        "outputs": {
            "claim_ledger": str(ledger_path),
            "claim_validation": str(validation_path),
            "table_inventory": str(inventory_path),
        },
        "checks": {
            "claim_rows": int(len(ledger)),
            "validation_failures": int((validation["status"] == "fail").sum()),
            "blocked_claims": int((validation["status"] == "blocked").sum()),
            "inventory_rows": int(len(inventory)),
        },
        "status": status,
    }
    atomic_write_json(paths.manifests / "claim_ledger.json", manifest)
    atomic_write_text(paths.reports / "claim_ledger_report.md", report_text(manifest, ledger, validation))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--panel-run-id", required=True)
    parser.add_argument("--lgbm-run-id", required=True)
    parser.add_argument("--set-run-id", required=True, help="Set-model run id, or comma-separated run ids.")
    parser.add_argument("--factor-run-id", required=True)
    args = parser.parse_args()
    project_root = require_project_root(resolve_project_root(args.project_root), SCRATCH_PROJECT_ROOT)
    manifest = run_claim_ledger(
        project_root,
        args.run_id,
        args.panel_run_id,
        args.lgbm_run_id,
        parse_run_ids(args.set_run_id),
        args.factor_run_id,
    )
    print(f"status={manifest['status']}")
    print(f"manifest={project_root / 'runs' / args.run_id / 'manifests' / 'claim_ledger.json'}")
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
