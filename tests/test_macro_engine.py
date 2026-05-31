import tempfile
import unittest
from pathlib import Path

import pandas as pd

from segment_macro_betas.macro_engine import (
    add_configured_availability,
    load_series_catalog,
    missing_credentials,
    parse_period_to_date,
    read_env_file,
    standard_macro_frame,
)


class MacroEngineTests(unittest.TestCase):
    def test_read_env_file_strips_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("FRED_API_KEY='abc'\n# ignored\nEMPTY=\n", encoding="utf-8")
            values = read_env_file(path)
            self.assertEqual(values["FRED_API_KEY"], "abc")
            self.assertEqual(values["EMPTY"], "")

    def test_load_series_catalog_reads_public_config(self) -> None:
        catalog = load_series_catalog(Path.cwd())
        ids = {item["series_id"] for item in catalog}
        sources = {item["source"] for item in catalog}
        self.assertIn("FEDFUNDS", ids)
        self.assertTrue({"fred", "bls", "bea", "eia"}.issubset(sources))
        self.assertTrue(all(item["macro_area"] for item in catalog))

    def test_configured_availability_uses_month_end_lag(self) -> None:
        df = pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]})
        series = {"release_lag_days": 7, "timing": "configured_release_lag", "revision_safe": False}
        out = add_configured_availability(df, series)
        self.assertEqual(str(out.loc[0, "available_date"].date()), "2020-02-07")
        self.assertTrue(bool(out.loc[0, "lookahead_safe"]))
        self.assertFalse(bool(out.loc[0, "revision_safe"]))

    def test_period_parsing_handles_month_quarter_and_year(self) -> None:
        self.assertEqual(str(parse_period_to_date("M02", year="2020").date()), "2020-02-29")
        self.assertEqual(str(parse_period_to_date("2020Q2").date()), "2020-06-30")
        self.assertEqual(str(parse_period_to_date("2020").date()), "2020-12-31")

    def test_standard_frame_adds_required_columns(self) -> None:
        frame = pd.DataFrame({"date": ["2020-01-31"], "value": ["1.25"]})
        series = {
            "source": "bls",
            "series_id": "LNS14000000",
            "series_name": "bls_unemployment_rate",
            "macro_area": "GLOBAL",
            "release_lag_days": 7,
            "timing": "configured_release_lag",
            "revision_safe": False,
        }
        out = standard_macro_frame(frame, series)
        self.assertEqual(set(out["source"]), {"bls"})
        self.assertIn("available_date", out.columns)
        self.assertIn("realtime_start", out.columns)
        self.assertAlmostEqual(float(out.loc[0, "value"]), 1.25)

    def test_missing_credentials_is_source_specific(self) -> None:
        catalog = [
            {"source": "fred"},
            {"source": "bls"},
            {"source": "bea"},
            {"source": "eia"},
        ]
        missing = missing_credentials(catalog, {"FRED_API_KEY": "x", "BLS_API_KEY": "x"})
        self.assertEqual(missing, ["BEA_API_KEY", "EIA_API_KEY"])


if __name__ == "__main__":
    unittest.main()
