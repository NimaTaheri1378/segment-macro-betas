import unittest

import pandas as pd

from segment_macro_betas.segment_set_model import (
    build_geo_vocab,
    build_panel_frame,
    encode_panel_sets,
    normalize_geo_label,
    parse_device_type,
    parse_variants,
    prepare_segment_tokens,
)


class SegmentSetModelTests(unittest.TestCase):
    def test_prepare_segment_tokens_computes_revenue_shares(self) -> None:
        raw = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "srcdate": ["2020-01-31", "2020-01-31"],
                "gareag": ["USA", "CAN"],
                "gareat": ["ISO", "ISO"],
                "sales": [60.0, 40.0],
                "revts": [None, None],
                "ias": [None, None],
                "sid": ["1", "2"],
            }
        )
        tokens = prepare_segment_tokens(raw)
        self.assertEqual(len(tokens), 2)
        self.assertAlmostEqual(float(tokens["revenue_share"].sum()), 1.0)

    def test_vocab_and_encoding_match_panel_rows(self) -> None:
        tokens = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "segment_srcdate": pd.to_datetime(["2020-01-31", "2020-01-31"]),
                "geo_label": ["USA", "CAN"],
                "revenue_share": [0.6, 0.4],
            }
        )
        frame = pd.DataFrame({"gvkey": ["001"], "segment_srcdate": pd.to_datetime(["2020-01-31"])})
        vocab = build_geo_vocab(tokens, 10)
        geo_ids, shares, checks = encode_panel_sets(frame, tokens, vocab, max_segments=3)
        self.assertEqual(geo_ids.shape, (1, 3))
        self.assertEqual(shares.shape, (1, 3))
        self.assertEqual(checks["matched_set_rows"], 1)
        self.assertAlmostEqual(float(shares[0, :2].sum()), 1.0)

    def test_build_panel_frame_derives_controls(self) -> None:
        panel = pd.DataFrame(
            {
                "gvkey": ["001"],
                "permno": [10001],
                "date": ["2020-01-31"],
                "segment_srcdate": ["2019-03-01"],
                "next_month_excess_ret": [0.01],
                "mktcap": [1000.0],
                "at": [500.0],
                "ceq": [250.0],
                "sale": [800.0],
                "ni": [50.0],
                "capx": [20.0],
                "xrd": [5.0],
                "dltt": [100.0],
                "dlc": [10.0],
            }
        )
        frame, controls = build_panel_frame(panel)
        self.assertEqual(len(frame), 1)
        self.assertIn("book_to_market", controls)
        self.assertAlmostEqual(float(frame.iloc[0]["book_to_market"]), 0.25)

    def test_parse_variants_and_label_normalization(self) -> None:
        self.assertEqual(parse_variants("set_only,set_plus_controls"), ["set_only", "set_plus_controls"])
        with self.assertRaises(ValueError):
            parse_variants("not_a_variant")
        self.assertEqual(normalize_geo_label(" us "), "US")

    def test_parse_device_type(self) -> None:
        self.assertEqual(parse_device_type(None), "auto")
        self.assertEqual(parse_device_type("CUDA"), "cuda")
        with self.assertRaises(ValueError):
            parse_device_type("tpu")


if __name__ == "__main__":
    unittest.main()
