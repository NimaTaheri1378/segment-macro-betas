import tempfile
import unittest
from pathlib import Path

import pandas as pd

from segment_macro_betas.publication_tables import (
    build_factor_alpha_table,
    build_model_comparison,
    display_frame,
    parse_run_ids,
    run_publication_tables,
    table_notes,
    validate_publication_tables,
    DISPLAY_FACTOR_COLUMNS,
)


class PublicationTablesTests(unittest.TestCase):
    def test_model_table_combines_lgbm_and_set_summaries(self) -> None:
        lgbm = pd.DataFrame(
            {
                "variant": ["segment_only"],
                "prediction_rows": [100],
                "mean_rank_ic": [0.04],
                "t_rank_ic": [3.1],
                "mean_q5_minus_q1": [0.002],
                "t_q5_minus_q1": [1.0],
            }
        )
        sets = pd.DataFrame(
            {
                "variant": ["set_only", "set_transformer"],
                "architecture": [None, "set_transformer"],
                "prediction_rows": [100, 100],
                "mean_rank_ic": [0.01, 0.007],
                "t_rank_ic": [0.8, 1.9],
                "mean_q5_minus_q1": [0.001, 0.0004],
                "t_q5_minus_q1": [0.4, 0.3],
            }
        )
        out = build_model_comparison(lgbm, sets)
        self.assertEqual(set(out["model_family"]), {"LightGBM", "Deep Sets"})
        self.assertIn("review_note", out.columns)
        self.assertEqual(out.loc[out["variant"] == "segment_only", "review_note"].iloc[0], "positive rank diagnostic")
        self.assertEqual(out.loc[out["variant"] == "set_only", "architecture"].iloc[0], "deep_sets")

    def test_parse_run_ids_accepts_comma_separated_values(self) -> None:
        self.assertEqual(parse_run_ids("set_a,set_b"), ["set_a", "set_b"])
        with self.assertRaises(ValueError):
            parse_run_ids(" , ")

    def test_factor_table_formats_review_notes(self) -> None:
        factor = pd.DataFrame(
            {
                "model_family": ["lgbm"],
                "variant": ["non_segment_controls"],
                "months": [191],
                "mean_gross_q5_minus_q1": [0.007],
                "t_gross_q5_minus_q1": [2.2],
                "mean_net_q5_minus_q1": [0.006],
                "t_net_q5_minus_q1": [2.1],
                "mean_turnover": [0.4],
                "gross_alpha": [0.005],
                "gross_alpha_t": [2.4],
                "net_alpha": [0.004],
                "net_alpha_t": [1.9],
                "alpha_months": [190],
            }
        )
        out = build_factor_alpha_table(factor)
        self.assertEqual(out.iloc[0]["model_family"], "LightGBM")
        self.assertEqual(out.iloc[0]["review_note"], "positive alpha and net-spread diagnostic")
        display = display_frame(out, DISPLAY_FACTOR_COLUMNS)
        self.assertEqual(display.iloc[0]["Months"], "191")
        self.assertEqual(display.iloc[0]["Gross alpha"], "0.0050")

    def test_validation_requires_guardrail_notes(self) -> None:
        model = pd.DataFrame({"model_family": ["LightGBM"], "variant": ["a"]})
        factor = pd.DataFrame({"model_family": ["LightGBM"], "variant": ["a"]})
        claim_validation = pd.DataFrame({"status": ["pass", "blocked"]})
        checks = validate_publication_tables(
            model,
            factor,
            claim_validation,
            table_notes(panel_run_id="panel", cost_bps=10, nw_lag=6),
        )
        self.assertFalse((checks["status"] == "fail").any())

    def test_run_publication_tables_writes_private_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = root / "artifacts" / "tables"
            for run_id in ["lgbm", "set", "factor", "claim"]:
                (tables / run_id).mkdir(parents=True)
            pd.DataFrame(
                {
                    "variant": ["segment_only"],
                    "prediction_rows": [10],
                    "mean_rank_ic": [0.03],
                    "t_rank_ic": [2.5],
                    "mean_q5_minus_q1": [0.001],
                    "t_q5_minus_q1": [0.5],
                }
            ).to_csv(tables / "lgbm" / "lgbm_summary.csv", index=False)
            pd.DataFrame(
                {
                    "variant": ["set_only"],
                    "architecture": ["deep_sets"],
                    "prediction_rows": [10],
                    "mean_rank_ic": [0.01],
                    "t_rank_ic": [1.1],
                    "mean_q5_minus_q1": [0.001],
                    "t_q5_minus_q1": [0.4],
                }
            ).to_csv(tables / "set" / "deepsets_summary.csv", index=False)
            pd.DataFrame(
                {
                    "model_family": ["lgbm"],
                    "variant": ["segment_only"],
                    "months": [12],
                    "mean_gross_q5_minus_q1": [0.001],
                    "t_gross_q5_minus_q1": [0.5],
                    "mean_net_q5_minus_q1": [0.001],
                    "t_net_q5_minus_q1": [0.4],
                    "mean_turnover": [0.2],
                    "gross_alpha": [0.001],
                    "gross_alpha_t": [0.3],
                    "net_alpha": [0.001],
                    "net_alpha_t": [0.2],
                    "alpha_months": [12],
                }
            ).to_csv(tables / "factor" / "factor_robustness_summary.csv", index=False)
            pd.DataFrame({"status": ["pass", "blocked"]}).to_csv(tables / "claim" / "claim_validation.csv", index=False)
            manifest = run_publication_tables(
                root,
                "pub",
                panel_run_id="panel",
                lgbm_run_id="lgbm",
                set_run_ids=["set"],
                factor_run_id="factor",
                claim_run_id="claim",
                cost_bps=10,
                nw_lag=6,
            )
            self.assertEqual(manifest["status"], "ok")
            self.assertTrue((tables / "pub" / "publication_model_comparison.csv").exists())
            self.assertTrue((root / "runs" / "pub" / "reports" / "publication_factor_alpha.md").exists())


if __name__ == "__main__":
    unittest.main()
