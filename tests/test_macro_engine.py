import tempfile
import unittest
from pathlib import Path

import pandas as pd

from segment_macro_betas.macro_engine import add_configured_availability, load_series_catalog, read_env_file


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
        self.assertIn("FEDFUNDS", ids)
        self.assertTrue(all(item["source"] == "fred" for item in catalog))
        self.assertTrue(all(item["macro_area"] for item in catalog))

    def test_configured_availability_uses_month_end_lag(self) -> None:
        df = pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]})
        series = {"release_lag_days": 7, "timing": "configured_release_lag", "revision_safe": False}
        out = add_configured_availability(df, series)
        self.assertEqual(str(out.loc[0, "available_date"].date()), "2020-02-07")
        self.assertTrue(bool(out.loc[0, "lookahead_safe"]))
        self.assertFalse(bool(out.loc[0, "revision_safe"]))


if __name__ == "__main__":
    unittest.main()
