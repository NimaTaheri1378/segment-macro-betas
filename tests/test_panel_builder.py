import unittest

import pandas as pd

from segment_macro_betas.panel_builder import apply_segment_activation_dates, clean_segments, prepare_crsp, prepare_funda


class PanelBuilderTests(unittest.TestCase):
    def test_clean_segments_aggregates_snapshot(self) -> None:
        raw = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "srcdate": ["2020-01-31", "2020-01-31"],
                "datadate": ["2020-01-31", "2020-01-31"],
                "gareat": ["ISO", "REG"],
                "gareag": ["USA", "EUROPE"],
                "sales": [60.0, 40.0],
                "revts": [None, None],
                "ias": [None, None],
                "sid": ["1", "2"],
            }
        )
        out = clean_segments(raw)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["domestic_share"]), 0.6)
        self.assertAlmostEqual(float(out.iloc[0]["foreign_share"]), 0.4)
        self.assertEqual(str(out.iloc[0]["segment_datadate"].date()), "2020-01-31")

    def test_activation_prefers_filing_date(self) -> None:
        snapshots = pd.DataFrame(
            {
                "gvkey": ["001", "002"],
                "segment_datadate": ["2020-01-31", "2020-01-31"],
                "segment_srcdate": ["2020-02-15", "2020-02-15"],
            }
        )
        filings = pd.DataFrame(
            {
                "gvkey": ["001"],
                "datadate": ["2020-01-31"],
                "filing_date": ["2020-03-01"],
                "filing_date_source": ["fdate"],
            }
        )
        out, checks = apply_segment_activation_dates(snapshots, filings, activation_lag_days=1)
        self.assertEqual(str(out.loc[out["gvkey"] == "001", "segment_activation_date"].iloc[0].date()), "2020-03-02")
        self.assertEqual(out.loc[out["gvkey"] == "001", "segment_activation_source"].iloc[0], "fdate")
        self.assertEqual(out.loc[out["gvkey"] == "002", "segment_activation_source"].iloc[0], "srcdate_fallback")
        self.assertAlmostEqual(checks["filing_date_match_rate"], 0.5)

    def test_prepare_funda_prefers_public_dates(self) -> None:
        raw = pd.DataFrame(
            {
                "gvkey": ["001", "002"],
                "datadate": ["2020-01-31", "2020-01-31"],
                "pdate": ["2020-02-20", None],
                "fdate": ["2020-03-01", "2020-03-15"],
                "at": [1.0, 2.0],
            }
        )
        out = prepare_funda(raw)
        self.assertEqual(str(out.loc[out["gvkey"] == "001", "funda_avail_date"].iloc[0].date()), "2020-02-20")
        self.assertEqual(out.loc[out["gvkey"] == "002", "funda_avail_source"].iloc[0], "fdate")

    def test_prepare_crsp_deduplicates_permno_month(self) -> None:
        raw = pd.DataFrame({"permno": [1, 1], "date": ["2020-01-31", "2020-01-31"], "ret": [0.1, 0.2]})
        out = prepare_crsp(raw)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["ret"]), 0.2)


if __name__ == "__main__":
    unittest.main()
