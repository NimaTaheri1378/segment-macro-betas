import unittest

import pandas as pd

from segment_macro_betas.filing_dates import prepare_filing_dates, years_from_args


class FilingDatesTests(unittest.TestCase):
    def test_years_from_args_accepts_ranges(self) -> None:
        self.assertEqual(years_from_args("2019,2021-2022", 2000, 2001), [2019, 2021, 2022])

    def test_prepare_filing_dates_uses_preliminary_then_file_date(self) -> None:
        raw = pd.DataFrame(
            {
                "gvkey": ["001", "002"],
                "datadate": ["2020-01-31", "2020-01-31"],
                "pdate": ["2020-02-15", None],
                "fdate": ["2020-03-01", "2020-03-10"],
                "fyear": [2020, 2020],
                "fyr": [1, 1],
            }
        )
        out = prepare_filing_dates(raw)
        self.assertEqual(str(out.loc[out["gvkey"] == "001", "filing_date"].iloc[0].date()), "2020-02-15")
        self.assertEqual(out.loc[out["gvkey"] == "001", "filing_date_source"].iloc[0], "pdate")
        self.assertEqual(out.loc[out["gvkey"] == "002", "filing_date_source"].iloc[0], "fdate")


if __name__ == "__main__":
    unittest.main()
