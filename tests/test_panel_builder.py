import pandas as pd
import unittest

from segment_macro_betas.panel_builder import clean_segments, prepare_crsp


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

    def test_prepare_crsp_deduplicates_permno_month(self) -> None:
        raw = pd.DataFrame({"permno": [1, 1], "date": ["2020-01-31", "2020-01-31"], "ret": [0.1, 0.2]})
        out = prepare_crsp(raw)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["ret"]), 0.2)


if __name__ == "__main__":
    unittest.main()
