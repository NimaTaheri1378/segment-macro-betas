import tempfile
import unittest
from pathlib import Path

import pandas as pd

from segment_macro_betas.macro_engine import (
    add_configured_availability,
    fred_query_params,
    load_series_catalog,
    macro_fetch_error,
    missing_credentials,
    normalise_series_config,
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

    def test_fred_realtime_availability_uses_realtime_start(self) -> None:
        df = pd.DataFrame({"date": ["2020-01-01"], "value": [1.0], "realtime_start": ["2020-02-07"]})
        series = {"release_lag_days": 99, "timing": "fred_initial_release", "revision_safe": True}
        out = add_configured_availability(df, series)
        self.assertEqual(str(out.loc[0, "available_date"].date()), "2020-02-07")
        self.assertEqual(out.loc[0, "timing_source"], "fred_initial_release")
        self.assertTrue(bool(out.loc[0, "revision_safe"]))

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

    def test_fred_realtime_config_defaults_to_revision_safe(self) -> None:
        series = normalise_series_config(
            {
                "source": "fred",
                "series_id": "UNRATE",
                "timing": "fred_initial_release",
            }
        )
        self.assertTrue(series["revision_safe"])
        params = fred_query_params(series, "key", "2020-01-01", "2020-12-31")
        self.assertEqual(params["output_type"], 4)
        self.assertEqual(params["realtime_start"], "1776-07-04")
        self.assertEqual(params["realtime_end"], "9999-12-31")

    def test_missing_credentials_is_source_specific(self) -> None:
        catalog = [
            {"source": "fred"},
            {"source": "bls"},
            {"source": "bea"},
            {"source": "eia"},
        ]
        missing = missing_credentials(catalog, {"FRED_API_KEY": "x", "BLS_API_KEY": "x"})
        self.assertEqual(missing, ["BEA_API_KEY", "EIA_API_KEY"])

    def test_macro_fetch_error_summarizes_without_credentials(self) -> None:
        series = {"source": "fred", "series_id": "FEDFUNDS", "series_name": "federal_funds_rate"}
        out = macro_fetch_error(series, RuntimeError("HTTP Error 429: Too Many Requests"))
        self.assertEqual(out["source"], "fred")
        self.assertEqual(out["series_id"], "FEDFUNDS")
        self.assertIn("429", out["message"])
        self.assertNotIn("API_KEY", str(out))


if __name__ == "__main__":
    unittest.main()
