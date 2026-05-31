import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from segment_macro_betas.visual_pack import build_model_comparison, fmt, parse_run_ids, sic_sector, write_dashboard


class VisualPackTests(unittest.TestCase):
    def test_sic_sector_maps_major_groups(self) -> None:
        self.assertEqual(sic_sector(2834), "Manufacturing")
        self.assertEqual(sic_sector("6021"), "Finance")
        self.assertEqual(sic_sector(None), "Unknown")

    def test_formatting_handles_missing_and_numbers(self) -> None:
        self.assertEqual(fmt(None), "n/a")
        self.assertEqual(fmt(1234), "1,234")
        self.assertEqual(fmt(0.123456, 2), "0.12")

    def test_model_comparison_combines_families(self) -> None:
        lgbm = pd.DataFrame(
            {
                "variant": ["all"],
                "prediction_rows": [10],
                "rank_ic_months": [2],
                "mean_rank_ic": [0.1],
                "t_rank_ic": [1.0],
                "mean_q5_minus_q1": [0.02],
                "t_q5_minus_q1": [1.2],
            }
        )
        deep = pd.DataFrame(
            {
                "variant": ["set_only", "set_transformer"],
                "architecture": [None, "set_transformer"],
                "prediction_rows": [8, 8],
                "rank_ic_months": [2, 2],
                "mean_rank_ic": [0.05, 0.01],
                "t_rank_ic": [0.8, 0.2],
                "mean_q5_minus_q1": [0.01, 0.001],
                "t_q5_minus_q1": [0.4, 0.1],
            }
        )
        out = build_model_comparison(lgbm, deep)
        self.assertEqual(set(out["family"]), {"LightGBM", "Deep Sets"})
        self.assertEqual(len(out), 3)

    def test_parse_run_ids(self) -> None:
        self.assertEqual(parse_run_ids("a,b"), ["a", "b"])
        with self.assertRaises(ValueError):
            parse_run_ids("")

    def test_dashboard_uses_relative_static_figure_paths(self) -> None:
        comparison = pd.DataFrame({"variant": ["all"], "mean_rank_ic": [0.1]})
        firm_explorer = pd.DataFrame({"permno": [10001], "foreign_sales_share": [0.2]})
        with TemporaryDirectory() as tmp:
            html_path = write_dashboard(
                Path(tmp),
                "run_1",
                {"sample_model_coverage": Path("/private/artifacts/figures_static/run_1/sample.png")},
                comparison,
                firm_explorer,
            )
            self.assertIn('src="../../figures_static/run_1/sample.png"', html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
