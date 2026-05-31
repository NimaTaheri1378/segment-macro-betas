import pandas as pd
import unittest

from segment_macro_betas.lgbm_benchmark import build_feature_frame, make_yearly_folds, monthly_rank_ic, parse_variants, select_features


class LgbmBenchmarkTests(unittest.TestCase):
    def test_build_feature_frame_creates_public_features(self) -> None:
        panel = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "permno": [10001, 10001],
                "date": ["2020-01-31", "2020-02-29"],
                "next_month_excess_ret": [0.01, -0.02],
                "foreign_share": [0.4, 0.5],
                "domestic_share": [0.6, 0.5],
                "geo_hhi": [0.52, 0.50],
                "geo_count": [2, 2],
                "top_geo_share": [0.6, 0.5],
                "segment_sales_sum": [100.0, 110.0],
                "mktcap": [1000.0, 1100.0],
                "at": [500.0, 520.0],
                "ceq": [250.0, 260.0],
                "sale": [800.0, 820.0],
                "ni": [50.0, 55.0],
                "capx": [20.0, 21.0],
                "xrd": [5.0, None],
                "dltt": [100.0, 105.0],
                "dlc": [10.0, 11.0],
            }
        )
        frame, features = build_feature_frame(panel)
        self.assertEqual(len(frame), 2)
        self.assertIn("book_to_market", features)
        self.assertIn("rd_to_assets", features)
        self.assertAlmostEqual(float(frame.iloc[0]["book_to_market"]), 0.25)
        self.assertTrue(all(str(frame[col].dtype) == "float64" for col in features + ["next_month_excess_ret"]))

    def test_make_yearly_folds_respects_min_train_and_holdout(self) -> None:
        dates = pd.date_range("2019-01-31", "2026-03-31", freq="ME")
        folds = make_yearly_folds(pd.Series(dates), min_train_months=24)
        self.assertEqual(folds[0]["fold_year"], 2021)
        self.assertLess(folds[-1]["validation_end"], pd.Timestamp("2026-01-01"))

    def test_parse_and_select_variants(self) -> None:
        variants = parse_variants("all,segment_only")
        self.assertEqual(variants, ["all", "segment_only"])
        selected = select_features(["foreign_share", "log_mktcap", "mktrf"], "segment_only")
        self.assertEqual(selected, ["foreign_share"])
        with self.assertRaises(ValueError):
            parse_variants("unknown")

    def test_monthly_rank_ic_skips_tiny_months(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": ["2020-01-31"] * 10,
                "prediction": range(10),
                "next_month_excess_ret": range(10),
            }
        )
        self.assertTrue(monthly_rank_ic(predictions).empty)


if __name__ == "__main__":
    unittest.main()
