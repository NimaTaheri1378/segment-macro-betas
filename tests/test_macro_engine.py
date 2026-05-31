import tempfile
import unittest
from pathlib import Path

from segment_macro_betas.macro_engine import read_env_file


class MacroEngineTests(unittest.TestCase):
    def test_read_env_file_strips_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("FRED_API_KEY='abc'\n# ignored\nEMPTY=\n", encoding="utf-8")
            values = read_env_file(path)
            self.assertEqual(values["FRED_API_KEY"], "abc")
            self.assertEqual(values["EMPTY"], "")


if __name__ == "__main__":
    unittest.main()
