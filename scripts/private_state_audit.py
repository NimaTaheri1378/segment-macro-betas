from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_RUNS = {
    "schema": ("20260530T222000Z", "schema_audit"),
    "full_extract": ("20260530T233446Z", "full_extract"),
    "panel": ("20260531T003936Z_panel_filing", "monthly_panel"),
    "baseline": ("20260531T004841Z_baseline_filing", "baselines"),
    "macro_nonfred": ("20260531T_macro_nonfred_full", "macro_engine"),
    "macro_tensor": ("20260531T_macro_tensor_nonfred_v2", "macro_tensor"),
    "lgbm_macro": ("20260531T_lgbm_macro_nonfred_v3", "lgbm_benchmark"),
    "deepsets": ("20260531T010832Z_set", "segment_set_model"),
    "set_transformer": ("20260531T_set_transformer_full", "segment_set_model"),
    "factor": ("20260531T_factor_robustness_macro_nonfred", "factor_robustness"),
    "claim_ledger": ("20260531T_claim_ledger_macro_nonfred", "claim_ledger"),
    "publication_tables": ("20260531T_publication_tables_macro_nonfred", "publication_tables"),
    "visual_pack": ("20260531T_visual_pack_macro_nonfred_v2", "visual_pack"),
    "fred_guard": ("20260531T_macro_full_catalog_guarded_smoke", "macro_engine"),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_manifest(root: Path, key: str) -> tuple[Path, dict[str, Any] | None]:
    run_id, name = EXPECTED_RUNS[key]
    path = root / "runs" / run_id / "manifests" / f"{name}.json"
    if not path.exists():
        return path, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def add(rows: list[dict[str, Any]], area: str, check: str, status: str, detail: str) -> None:
    rows.append({"area": area, "check": check, "status": status, "detail": detail})


def require(rows: list[dict[str, Any]], area: str, check: str, condition: bool, detail: str) -> None:
    add(rows, area, check, "pass" if condition else "fail", detail)


def load_all(root: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for key in EXPECTED_RUNS:
        path, manifest = read_manifest(root, key)
        if manifest is None:
            add(rows, key, "manifest_exists", "fail", str(path))
        else:
            manifests[key] = manifest
            add(rows, key, "manifest_exists", "pass", str(path.relative_to(root)))
    return manifests


def audit(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    manifests = load_all(root, rows)

    schema = manifests.get("schema", {})
    roles = schema.get("roles", {})
    require(rows, "schema", "wrds_connection", nested(schema, "wrds", "connection_status") == "ok", str(schema.get("wrds", {})))
    require(rows, "schema", "wrds_smoke_test", nested(schema, "wrds", "smoke_test_ok") is True, str(schema.get("wrds", {})))
    require(rows, "schema", "no_schema_errors", schema.get("errors") == [], f"errors={schema.get('errors')}")
    selected_ok = bool(roles) and all(role.get("selected") for role in roles.values())
    missing_groups_ok = bool(roles) and all(not nested(role, "selected", "missing_required_groups", default=["missing"]) for role in roles.values())
    require(rows, "schema", "selected_tables", selected_ok, f"roles={len(roles)}")
    require(rows, "schema", "required_columns_resolved", missing_groups_ok, f"roles={len(roles)}")

    full = manifests.get("full_extract", {})
    years = full.get("years", [])
    require(rows, "full_extract", "status_ok", full.get("status") == "ok" and full.get("execute") is True, f"status={full.get('status')} execute={full.get('execute')}")
    require(rows, "full_extract", "development_years", years == list(range(2006, 2026)), f"years={years[:2]}..{years[-2:] if years else []} n={len(years)}")
    require(rows, "full_extract", "annual_shards_present", len(full.get("shards", {})) == 20, f"shard_years={len(full.get('shards', {}))}")

    panel = manifests.get("panel", {})
    require(rows, "panel", "status_ok", panel.get("status") == "ok", f"status={panel.get('status')}")
    require(rows, "panel", "row_count", nested(panel, "checks", "panel_rows", default=0) >= 900_000, f"panel_rows={nested(panel, 'checks', 'panel_rows')}")
    require(rows, "panel", "activation_no_violations", nested(panel, "checks", "activation_rule_violations") == 0, f"violations={nested(panel, 'checks', 'activation_rule_violations')}")
    require(rows, "panel", "development_window", nested(panel, "checks", "max_date") == "2025-12-31", f"max_date={nested(panel, 'checks', 'max_date')}")

    baseline = manifests.get("baseline", {})
    require(rows, "baseline", "status_ok", baseline.get("status") == "ok", f"status={baseline.get('status')}")
    require(rows, "baseline", "rank_ic_months", nested(baseline, "checks", "rank_ic_months", default=0) >= 200, f"rank_ic_months={nested(baseline, 'checks', 'rank_ic_months')}")

    macro = manifests.get("macro_nonfred", {})
    require(rows, "macro_nonfred", "status_ok", macro.get("status") == "ok", f"status={macro.get('status')}")
    require(rows, "macro_nonfred", "official_sources", nested(macro, "checks", "source_count", default=0) >= 3, f"sources={nested(macro, 'checks', 'sources')}")
    require(rows, "macro_nonfred", "lookahead_safe", nested(macro, "checks", "lookahead_safe") is True, f"lookahead_safe={nested(macro, 'checks', 'lookahead_safe')}")
    if nested(macro, "checks", "revision_safe") is not True:
        add(rows, "macro_nonfred", "revision_safe", "blocked", "non-FRED official run uses configured release lags, not true revision vintages")

    tensor = manifests.get("macro_tensor", {})
    tensor_macro = nested(tensor, "checks", "macro", default={})
    require(rows, "macro_tensor", "status_ok", tensor.get("status") == "ok", f"status={tensor.get('status')}")
    require(rows, "macro_tensor", "coverage", nested(tensor, "checks", "aggregation", "macro_coverage_rate", default=0) >= 0.999, f"coverage={nested(tensor, 'checks', 'aggregation', 'macro_coverage_rate')}")
    require(rows, "macro_tensor", "feature_count", nested(tensor, "checks", "aggregation", "macro_feature_count", default=0) >= 9, f"features={nested(tensor, 'checks', 'aggregation', 'macro_feature_count')}")
    require(rows, "macro_tensor", "joined_match", nested(tensor, "checks", "joined_token_match_rate", default=0) >= 0.999, f"match={nested(tensor, 'checks', 'joined_token_match_rate')}")
    require(rows, "macro_tensor", "lookahead_safe", tensor_macro.get("lookahead_safe") is True, f"macro={tensor_macro}")
    if tensor_macro.get("revision_safe") is not True:
        add(rows, "macro_tensor", "revision_safe", "blocked", "tensor source is no-lookahead but not revision-safe")

    lgbm = manifests.get("lgbm_macro", {})
    require(rows, "lgbm_macro", "status_ok", lgbm.get("status") == "ok", f"status={lgbm.get('status')}")
    require(rows, "lgbm_macro", "variants", nested(lgbm, "checks", "variants_ok", default=0) >= 4, f"checks={lgbm.get('checks')}")
    require(rows, "lgbm_macro", "macro_only_diagnostic", nested(lgbm, "checks", "variants_diagnostic_only") == 1, f"checks={lgbm.get('checks')}")
    require(rows, "lgbm_macro", "best_spread_macro", nested(lgbm, "checks", "best_spread_variant") == "all_plus_macro", f"checks={lgbm.get('checks')}")

    for key in ["deepsets", "set_transformer"]:
        model = manifests.get(key, {})
        require(rows, key, "status_ok", model.get("status") == "ok", f"status={model.get('status')}")
        require(rows, key, "variant_count", nested(model, "checks", "variants_ok", default=0) >= 1, f"checks={model.get('checks')}")

    factor = manifests.get("factor", {})
    require(rows, "factor", "status_ok", factor.get("status") == "ok", f"status={factor.get('status')}")
    require(rows, "factor", "variants", nested(factor, "checks", "variants", default=0) >= 7, f"checks={factor.get('checks')}")
    require(rows, "factor", "factor_months", nested(factor, "checks", "factor_months", default=0) >= 200, f"checks={factor.get('checks')}")

    claim = manifests.get("claim_ledger", {})
    require(rows, "claim_ledger", "status_ok", claim.get("status") == "ok", f"status={claim.get('status')}")
    require(rows, "claim_ledger", "no_validation_failures", nested(claim, "checks", "validation_failures") == 0, f"checks={claim.get('checks')}")
    if nested(claim, "checks", "blocked_claims", default=0) > 0:
        add(rows, "claim_ledger", "blocked_claims", "blocked", f"blocked_claims={nested(claim, 'checks', 'blocked_claims')}")

    pub = manifests.get("publication_tables", {})
    require(rows, "publication_tables", "status_ok", pub.get("status") == "ok", f"status={pub.get('status')}")
    require(rows, "publication_tables", "review_clean", nested(pub, "checks", "review_failures") == 0, f"checks={pub.get('checks')}")
    require(rows, "publication_tables", "claim_validation_clean", nested(pub, "checks", "claim_validation_failures") == 0, f"checks={pub.get('checks')}")

    visual = manifests.get("visual_pack", {})
    require(rows, "visual_pack", "status_ok", visual.get("status") == "ok", f"status={visual.get('status')}")
    require(rows, "visual_pack", "figures", nested(visual, "checks", "figure_count", default=0) >= 7, f"checks={visual.get('checks')}")

    fred = manifests.get("fred_guard", {})
    errors = fred.get("api_errors", [])
    fred_429 = bool(errors and errors[0].get("http_status") == 429)
    require(rows, "fred_guard", "guarded_api_error", fred.get("status") == "api_error" and fred_429, f"status={fred.get('status')} errors={errors[:1]}")
    add(rows, "fred_guard", "full_fred_catalog", "blocked", "full FRED-inclusive catalog is rate-limited; do not retry aggressively")

    failures = [row for row in rows if row["status"] == "fail"]
    blockers = [row for row in rows if row["status"] == "blocked"]
    status = "fail" if failures else ("ok_with_blockers" if blockers else "ok")
    return {
        "created_utc": now_iso(),
        "status": status,
        "checks": {
            "passed": sum(row["status"] == "pass" for row in rows),
            "failed": len(failures),
            "blocked": len(blockers),
            "total": len(rows),
        },
        "rows": rows,
    }


def write_outputs(root: Path, run_id: str, result: dict[str, Any]) -> None:
    run_root = root / "runs" / run_id
    manifest_dir = run_root / "manifests"
    report_dir = run_root / "reports"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "private_state_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Private State Audit",
        "",
        f"- Status: `{result['status']}`",
        f"- Passed: `{result['checks']['passed']}`",
        f"- Failed: `{result['checks']['failed']}`",
        f"- Blocked: `{result['checks']['blocked']}`",
        "",
        "| Area | Check | Status | Detail |",
        "|---|---|---|---|",
    ]
    for row in result["rows"]:
        detail = str(row["detail"]).replace("|", "\\|")
        lines.append(f"| `{row['area']}` | `{row['check']}` | `{row['status']}` | {detail} |")
    lines.append("")
    (report_dir / "private_state_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", default=None, help="Optional ignored run folder for audit outputs.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when blockers remain.")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    result = audit(root)
    if args.run_id:
        write_outputs(root, args.run_id, result)
    print(f"private_state_audit_{result['status']}")
    print(json.dumps(result["checks"], sort_keys=True))
    if result["status"] == "fail":
        return 1
    if args.strict and result["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
