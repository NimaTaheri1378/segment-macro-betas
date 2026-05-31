import unittest

from segment_macro_betas.full_extract import years_from_args


class FullExtractTests(unittest.TestCase):
    def test_year_parser_supports_ranges(self) -> None:
        self.assertEqual(years_from_args("2006,2008-2010", 2000, 2001), [2006, 2008, 2009, 2010])

    def test_year_parser_default_window(self) -> None:
        self.assertEqual(years_from_args(None, 2006, 2008), [2006, 2007, 2008])


if __name__ == "__main__":
    unittest.main()
