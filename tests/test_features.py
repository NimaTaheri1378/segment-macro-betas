import unittest

from segment_macro_betas.features.exposures import hhi, is_domestic_label


class ExposureFeatureTests(unittest.TestCase):
    def test_domestic_label_detection(self) -> None:
        self.assertTrue(is_domestic_label("United States"))
        self.assertTrue(is_domestic_label("U.S."))
        self.assertFalse(is_domestic_label("Europe"))

    def test_hhi_normalizes_weights(self) -> None:
        self.assertAlmostEqual(hhi([0.5, 0.5]), 0.5)
        self.assertAlmostEqual(hhi([2.0, 2.0]), 0.5)
        self.assertEqual(hhi([]), 0.0)


if __name__ == "__main__":
    unittest.main()
