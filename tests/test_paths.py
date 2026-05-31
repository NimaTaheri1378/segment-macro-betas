from pathlib import Path
import tempfile
import unittest

from segment_macro_betas.paths import ensure_within, make_run_paths


class PathGuardTests(unittest.TestCase):
    def test_ensure_within_rejects_outside_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaises(ValueError):
                ensure_within(root, root.parent / "outside")

    def test_make_run_paths_stays_inside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            paths = make_run_paths(root, "test_run")
            self.assertTrue(str(paths.run_root).startswith(str(root)))
            self.assertTrue(paths.logs.exists())


if __name__ == "__main__":
    unittest.main()
