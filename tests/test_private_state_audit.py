import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.private_state_audit import EXPECTED_RUNS, audit


def write_manifest(root: Path, key: str, manifest: dict) -> None:
    run_id, name = EXPECTED_RUNS[key]
    path = root / "runs" / run_id / "manifests" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


class PrivateStateAuditTests(unittest.TestCase):
    def test_audit_reports_known_blockers_without_failures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_manifest(
                root,
                "schema",
                {
                    "wrds": {"connection_status": "ok", "smoke_test_ok": True},
                    "errors": [],
                    "roles": {
                        "segments": {"selected": {"missing_required_groups": []}},
                        "crsp": {"selected": {"missing_required_groups": []}},
                    },
                },
            )
            write_manifest(
                root,
                "full_extract",
                {"status": "ok", "execute": True, "years": list(range(2006, 2026)), "shards": {str(year): {} for year in range(2006, 2026)}},
            )
            write_manifest(
                root,
                "panel",
                {"status": "ok", "checks": {"panel_rows": 936897, "activation_rule_violations": 0, "max_date": "2025-12-31"}},
            )
            write_manifest(root, "baseline", {"status": "ok", "checks": {"rank_ic_months": 238}})
            write_manifest(
                root,
                "macro_nonfred",
                {"status": "ok", "checks": {"source_count": 3, "sources": ["bls", "bea", "eia"], "lookahead_safe": True, "revision_safe": False}},
            )
            write_manifest(
                root,
                "macro_tensor",
                {
                    "status": "ok",
                    "checks": {
                        "aggregation": {"macro_coverage_rate": 1.0, "macro_feature_count": 9},
                        "joined_token_match_rate": 1.0,
                        "macro": {"lookahead_safe": True, "revision_safe": False},
                    },
                },
            )
            write_manifest(
                root,
                "macro_fred_initial",
                {"status": "ok", "checks": {"series_count": 2, "lookahead_safe": True, "revision_safe": True}},
            )
            write_manifest(
                root,
                "macro_tensor_fred_initial",
                {
                    "status": "ok",
                    "checks": {
                        "aggregation": {"macro_coverage_rate": 1.0, "macro_feature_count": 6},
                        "macro": {"vintage_safe": True},
                    },
                },
            )
            write_manifest(
                root,
                "macro_full_catalog",
                {"status": "ok", "checks": {"series_count": 8, "source_count": 4, "lookahead_safe": True}},
            )
            write_manifest(
                root,
                "macro_tensor_full_catalog",
                {
                    "status": "ok",
                    "checks": {
                        "aggregation": {"macro_coverage_rate": 1.0, "macro_feature_count": 24},
                        "macro": {"availability_source": "available_date"},
                    },
                },
            )
            write_manifest(
                root,
                "lgbm_macro",
                {"status": "ok", "checks": {"variants_ok": 4, "variants_diagnostic_only": 1, "best_spread_variant": "all_plus_macro"}},
            )
            write_manifest(
                root,
                "lgbm_fred_initial",
                {"status": "ok", "checks": {"variants_ok": 4, "variants_diagnostic_only": 1, "best_spread_variant": "all_plus_macro"}},
            )
            write_manifest(
                root,
                "lgbm_full_catalog",
                {"status": "ok", "checks": {"variants_ok": 4, "variants_diagnostic_only": 1, "best_spread_variant": "all_plus_macro"}},
            )
            write_manifest(root, "deepsets", {"status": "ok", "checks": {"variants_ok": 2}})
            write_manifest(root, "set_transformer", {"status": "ok", "checks": {"variants_ok": 1}})
            write_manifest(root, "factor", {"status": "ok", "checks": {"variants": 7, "factor_months": 239}})
            write_manifest(root, "factor_fred_initial", {"status": "ok", "checks": {"variants": 7, "factor_months": 239}})
            write_manifest(root, "factor_full_catalog", {"status": "ok", "checks": {"variants": 7, "factor_months": 239}})
            write_manifest(root, "claim_ledger", {"status": "ok", "checks": {"validation_failures": 0, "blocked_claims": 1}})
            write_manifest(root, "claim_ledger_fred_initial", {"status": "ok", "checks": {"validation_failures": 0, "blocked_claims": 0}})
            write_manifest(root, "claim_ledger_full_catalog", {"status": "ok", "checks": {"validation_failures": 0, "blocked_claims": 0}})
            write_manifest(root, "publication_tables", {"status": "ok", "checks": {"review_failures": 0, "claim_validation_failures": 0}})
            write_manifest(root, "publication_tables_fred_initial", {"status": "ok", "checks": {"review_failures": 0, "claim_validation_failures": 0}})
            write_manifest(root, "publication_tables_full_catalog", {"status": "ok", "checks": {"review_failures": 0, "claim_validation_failures": 0}})
            write_manifest(root, "visual_pack", {"status": "ok", "checks": {"figure_count": 7}})
            write_manifest(root, "visual_pack_fred_initial", {"status": "ok", "checks": {"figure_count": 7}})
            write_manifest(root, "visual_pack_full_catalog", {"status": "ok", "checks": {"figure_count": 7}})
            write_manifest(
                root,
                "holdout_protocol",
                {
                    "status": "frozen",
                    "holdout_opened": False,
                    "selected_model": {"variant": "all_plus_macro"},
                    "checks": {"failed": 0},
                },
            )
            result = audit(root)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["checks"]["failed"], 0)
        self.assertEqual(result["checks"]["blocked"], 0)


if __name__ == "__main__":
    unittest.main()
