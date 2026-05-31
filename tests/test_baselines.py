import pandas as pd
import unittest

from segment_macro_betas.baselines import safe_quintiles, t_stat


class BaselineTests(unittest.TestCase):
    def test_safe_quintiles_assigns_five_buckets(self) -> None:
        q = safe_quintiles(pd.Series(range(50)))
        self.assertEqual(set(q.dropna().astype(int)), {1, 2, 3, 4, 5})

    def test_t_stat_handles_short_series(self) -> None:
        self.assertIsNone(t_stat(pd.Series([1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
