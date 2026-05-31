from pathlib import Path
import unittest


class PublicSafetyTests(unittest.TestCase):
    def test_no_env_file_committed(self) -> None:
        self.assertFalse(Path(".env").exists())

    def test_gitignore_protects_private_paths(self) -> None:
        text = Path(".gitignore").read_text(encoding="utf-8")
        for pattern in [".codex/", ".env", "data/", "runs/", "artifacts/private/", "artifacts/tables/", "*.egg-info/"]:
            self.assertIn(pattern, text)

    def test_env_example_has_no_values(self) -> None:
        for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#"):
                self.assertTrue(line.endswith("="), line)


if __name__ == "__main__":
    unittest.main()
