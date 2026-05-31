import tempfile
import unittest
from pathlib import Path

import pandas as pd

from segment_macro_betas.factor_robustness import (
    compute_spreads,
    compute_turnover,
    factor_alpha,
    load_model_predictions,
    parse_run_specs,
    prediction_quintile_panel,
    prepare_factor_returns,
    summarize_factor_robustness,
)


class FactorRobustnessTests(unittest.TestCase):
    def test_parse_run_specs(self) -> None:
        self.assertEqual(parse_run_specs("lgbm:a,deepsets:b"), [("lgbm", "a"), ("deepsets", "b")])
        with self.assertRaises(ValueError):
            parse_run_specs("bad")
        with self.assertRaises(ValueError):
            parse_run_specs("other:a")

    def test_load_model_predictions_from_private_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run1"
            run_dir.mkdir()
            pd.DataFrame(
                {
                    "gvkey": ["001"],
                    "permno": [10001],
                    "date": ["2020-01-31"],
                    "prediction": [0.5],
                    "next_month_excess_ret": [0.02],
                }
            ).to_parquet(run_dir / "lgbm_all_predictions.parquet")
            out = load_model_predictions(root, [("lgbm", "run1")])
            self.assertEqual(len(out), 1)
            self.assertEqual(out.iloc[0]["variant"], "all")
            self.assertEqual(out.iloc[0]["model_family"], "lgbm")

    def test_spreads_turnover_and_factor_alpha(self) -> None:
        rows = []
        for month_idx, date in enumerate(pd.date_range("2020-01-31", periods=14, freq="ME")):
            for i in range(50):
                rows.append(
                    {
                        "model_family": "lgbm",
                        "model_run_id": "run1",
                        "variant": "all",
                        "date": date,
                        "permno": 10000 + i,
                        "prediction": i + month_idx * 0.01,
                        "next_month_excess_ret": (i / 10000.0) + 0.001,
                    }
                )
        predictions = pd.DataFrame(rows)
        scored = prediction_quintile_panel(predictions)
        spreads = compute_spreads(scored)
        turnover = compute_turnover(scored)
        self.assertEqual(spreads["date"].nunique(), 14)
        self.assertIn("long_short_turnover", turnover.columns)

        factors = pd.DataFrame(
            {
                "date": pd.date_range("2020-02-29", periods=14, freq="ME"),
                "mktrf": [0.0] * 14,
                "smb": [0.0] * 14,
                "hml": [0.0] * 14,
                "umd": [0.0] * 14,
            }
        )
        summary, merged, turnover = summarize_factor_robustness(predictions, factors, cost_bps=10, nw_lag=3)
        self.assertEqual(len(summary), 1)
        self.assertGreater(float(summary.iloc[0]["gross_alpha"]), 0)
        self.assertIn("net_q5_minus_q1", merged.columns)

    def test_prepare_factor_returns_deduplicates_months(self) -> None:
        panel = pd.DataFrame({"date": ["2020-01-31", "2020-01-31"], "mktrf": [0.1, 0.2], "smb": [0.0, 0.0]})
        factors = prepare_factor_returns(panel)
        self.assertEqual(len(factors), 1)
        self.assertAlmostEqual(float(factors.iloc[0]["mktrf"]), 0.2)

    def test_factor_alpha_returns_none_for_short_series(self) -> None:
        out = factor_alpha(pd.Series([0.01, 0.02]), pd.DataFrame({"mktrf": [0.0, 0.0]}), nw_lag=1)
        self.assertIsNone(out["alpha"])


if __name__ == "__main__":
    unittest.main()
