from pathlib import Path
import subprocess
import unittest


class PublicSafetyTests(unittest.TestCase):
    def test_env_file_is_private_not_tracked(self) -> None:
        try:
            tracked = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True)
        except FileNotFoundError:
            tracked = None
        if tracked is not None and tracked.returncode == 0:
            self.assertEqual("", tracked.stdout.strip())
        self.assertIn(".env", Path(".gitignore").read_text(encoding="utf-8"))

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
