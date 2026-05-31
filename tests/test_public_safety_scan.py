import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.public_safety_scan import iter_public_files


class PublicSafetyScanTests(unittest.TestCase):
    def test_private_env_is_skipped_but_example_is_public(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("PRIVATE_ONLY=1", encoding="utf-8")
            (root / ".env.example").write_text("FRED_API_KEY=", encoding="utf-8")
            files = {path.name for path in iter_public_files(root)}
        self.assertNotIn(".env", files)
        self.assertIn(".env.example", files)


if __name__ == "__main__":
    unittest.main()
