import unittest
from pathlib import Path

import pandas as pd

from segment_macro_betas.claim_ledger import build_claim_ledger, build_table_inventory, macro_execution_claim, parse_run_ids, validate_claims


class ClaimLedgerTests(unittest.TestCase):
    def test_ledger_contains_guarded_claims_and_macro_blocker(self) -> None:
        lgbm = pd.DataFrame(
            [
                {
                    "variant": "segment_only",
                    "mean_rank_ic": 0.03,
                    "t_rank_ic": 4.0,
                    "mean_q5_minus_q1": 0.002,
                },
                {"variant": "non_segment_controls", "mean_q5_minus_q1": 0.007, "t_rank_ic": 2.0},
            ]
        )
        sets = pd.DataFrame(
            [
                {"variant": "set_only", "mean_rank_ic": 0.01, "t_rank_ic": 2.5, "mean_q5_minus_q1": 0.001},
                {
                    "variant": "set_transformer",
                    "model_run_id": "set_transformer",
                    "mean_rank_ic": 0.007,
                    "t_rank_ic": 1.9,
                    "mean_q5_minus_q1": 0.0004,
                },
            ]
        )
        factors = pd.DataFrame(
            [
                {
                    "model_family": "lgbm",
                    "variant": "no_return_or_market",
                    "gross_alpha": 0.006,
                    "gross_alpha_t": 2.6,
                    "mean_net_q5_minus_q1": 0.005,
                    "t_net_q5_minus_q1": 2.4,
                }
            ]
        )
        ledger = build_claim_ledger(
            panel_run_id="panel",
            lgbm_run_id="lgbm",
            set_run_id="set",
            factor_run_id="factor",
            lgbm_summary=lgbm,
            set_summary=sets,
            factor_summary=factors,
        )
        self.assertGreaterEqual(len(ledger), 6)
        self.assertTrue(ledger["allowed_wording"].str.contains("filing-date panel").any())
        self.assertTrue(ledger["allowed_wording"].str.contains("Set Transformer").any())
        self.assertTrue(ledger["evidence_strength"].str.contains("blocked").any())
        validation = validate_claims(ledger)
        self.assertEqual(int((validation["status"] == "fail").sum()), 0)
        self.assertGreaterEqual(int((validation["status"] == "blocked").sum()), 1)

    def test_validation_flags_forbidden_allowed_wording(self) -> None:
        ledger = pd.DataFrame(
            [
                {
                    "module": "x",
                    "allowed_wording": "This proves arbitrage.",
                    "evidence_strength": "passed diagnostic",
                }
            ]
        )
        validation = validate_claims(ledger)
        self.assertEqual(validation.loc[0, "status"], "fail")

    def test_macro_execution_claim_distinguishes_fred_initial_release(self) -> None:
        fred = macro_execution_claim("20260531T_lgbm_fred_initial_release")
        nonfred = macro_execution_claim("20260531T_lgbm_macro_nonfred_v3")
        self.assertEqual(fred["evidence_strength"], "passed diagnostic")
        self.assertIn("revision-safe for the included FRED series", fred["allowed_wording"])
        self.assertIn("blocked", nonfred["evidence_strength"])

    def test_parse_run_ids(self) -> None:
        self.assertEqual(parse_run_ids("set_a,set_b"), ["set_a", "set_b"])
        with self.assertRaises(ValueError):
            parse_run_ids("")

    def test_table_inventory_handles_missing_dirs(self) -> None:
        inventory = build_table_inventory({"missing": "nope"}, Path("."))
        self.assertEqual(len(inventory), 0)


if __name__ == "__main__":
    unittest.main()
