import unittest

import pandas as pd

from segment_macro_betas.macro_tensor import (
    build_macro_tensor,
    canonical_macro_area,
    prepare_macro_states,
    prepare_segment_geo_tokens,
)


class MacroTensorTests(unittest.TestCase):
    def test_canonical_macro_area_maps_common_labels(self) -> None:
        self.assertEqual(canonical_macro_area("global"), "GLOBAL")
        self.assertEqual(canonical_macro_area("United States"), "USA")
        self.assertEqual(canonical_macro_area("CHIN"), "CHINA")
        self.assertEqual(canonical_macro_area("Europe"), "EUROPE")
        self.assertEqual(canonical_macro_area(None), "UNKNOWN")

    def test_prepare_segment_geo_tokens_computes_revenue_shares(self) -> None:
        raw = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "srcdate": ["2020-01-31", "2020-01-31"],
                "datadate": ["2019-12-31", "2019-12-31"],
                "gareag": ["USA", "Europe"],
                "gareat": ["", ""],
                "sales": [70.0, 30.0],
                "revts": [None, None],
                "ias": [None, None],
                "sid": ["1", "2"],
            }
        )
        out = prepare_segment_geo_tokens(raw).sort_values("macro_area")
        self.assertEqual(set(out["macro_area"]), {"EUROPE", "USA"})
        self.assertAlmostEqual(float(out["revenue_share"].sum()), 1.0)

    def test_prepare_macro_states_prefers_realtime_availability(self) -> None:
        raw = pd.DataFrame(
            {
                "series_id": ["FEDFUNDS", "FEDFUNDS"],
                "date": ["2020-01-31", "2020-02-29"],
                "realtime_start": ["2020-02-10", "2020-03-10"],
                "value": [1.5, 1.25],
            }
        )
        states, checks = prepare_macro_states(raw)
        self.assertTrue(checks["vintage_safe"])
        self.assertEqual(checks["availability_source"], "realtime_start")
        self.assertIn("federal_funds_rate", set(states["series_name"]))

    def test_build_macro_tensor_asof_joins_without_future_macro_values(self) -> None:
        panel = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "permno": [10001, 10001],
                "date": ["2020-02-29", "2020-03-31"],
                "segment_srcdate": ["2020-01-31", "2020-01-31"],
                "next_month_excess_ret": [0.01, 0.02],
            }
        )
        segments = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "srcdate": ["2020-01-31", "2020-01-31"],
                "datadate": ["2019-12-31", "2019-12-31"],
                "gareag": ["USA", "Europe"],
                "gareat": ["", ""],
                "sales": [60.0, 40.0],
                "revts": [None, None],
                "ias": [None, None],
                "sid": ["1", "2"],
            }
        )
        macro = pd.DataFrame(
            {
                "series_id": ["FEDFUNDS", "FEDFUNDS", "EU_IP", "EU_IP"],
                "macro_area": ["USA", "USA", "EUROPE", "EUROPE"],
                "date": ["2020-01-31", "2020-03-31", "2020-01-31", "2020-03-31"],
                "available_date": ["2020-02-15", "2020-04-15", "2020-02-20", "2020-04-20"],
                "value": [2.0, 3.0, 10.0, 20.0],
            }
        )
        tensor, tokens, checks = build_macro_tensor(panel, segments, macro)
        first = tensor[tensor["date"] == pd.Timestamp("2020-02-29")].iloc[0]
        second = tensor[tensor["date"] == pd.Timestamp("2020-03-31")].iloc[0]
        self.assertAlmostEqual(float(first["segment_macro_federal_funds_rate"]), 1.2)
        self.assertAlmostEqual(float(first["segment_macro_eu_ip"]), 4.0)
        self.assertAlmostEqual(float(second["segment_macro_federal_funds_rate"]), 1.2)
        self.assertAlmostEqual(float(second["segment_macro_eu_ip"]), 4.0)
        self.assertEqual(len(tokens), 4)
        self.assertEqual(checks["aggregation"]["firm_month_rows"], 2)
        self.assertTrue(checks["macro"]["vintage_safe"])

    def test_global_macro_area_applies_to_all_segment_tokens(self) -> None:
        panel = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "permno": [10, 10],
                "date": ["2020-03-31", "2020-04-30"],
                "segment_srcdate": ["2019-12-31", "2019-12-31"],
            }
        )
        segments = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "srcdate": ["2019-12-31", "2019-12-31"],
                "datadate": ["2019-12-31", "2019-12-31"],
                "gareag": ["United States", "Europe"],
                "gareat": ["", ""],
                "sales": [60.0, 40.0],
                "revts": [None, None],
                "ias": [None, None],
                "sid": ["A", "B"],
            }
        )
        macro = pd.DataFrame(
            {
                "series_id": ["FEDFUNDS"],
                "series_name": ["federal_funds_rate"],
                "macro_area": ["GLOBAL"],
                "date": ["2020-01-31"],
                "available_date": ["2020-02-07"],
                "value": [1.5],
            }
        )
        tensor, tokens, checks = build_macro_tensor(panel, segments, macro)
        self.assertEqual(len(tokens), 4)
        self.assertEqual(float(tokens["macro_federal_funds_rate"].notna().mean()), 1.0)
        self.assertAlmostEqual(float(tensor.loc[0, "segment_macro_federal_funds_rate"]), 1.5)
        self.assertTrue(checks["macro"]["vintage_safe"])


if __name__ == "__main__":
    unittest.main()
