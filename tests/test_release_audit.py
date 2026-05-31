import subprocess
import sys
import unittest
import re
from pathlib import Path


class ReleaseAuditTests(unittest.TestCase):
    def test_release_audit_script_passes_current_tree(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/release_audit.py"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("release_audit_ok", result.stdout)

    def test_runner_helper_is_public_safe(self) -> None:
        text = Path("scripts/_amarel_env.sh").read_text(encoding="utf-8")
        self.assertIn("SMB_PROJECT_ROOT", text)
        self.assertIn("SMB_SLURM_JOB_ID", text)
        self.assertNotIn("/scratch/", text)
        self.assertIsNone(re.search(r'EXPECTED_JOB_ID="\d+', text))


if __name__ == "__main__":
    unittest.main()
