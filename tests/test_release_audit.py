import subprocess
import sys
import unittest
import re
from pathlib import Path

from scripts import release_audit


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

    def test_ci_runs_release_audit(self) -> None:
        text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/public_safety_scan.py", text)
        self.assertIn("scripts/release_audit.py", text)
        self.assertIn("python -m unittest discover -s tests", text)

    def test_release_audit_requires_public_scripts_and_docs(self) -> None:
        required = set(release_audit.REQUIRED_PUBLIC_FILES)
        self.assertIn("docs/output_inventory.md", required)
        self.assertIn("docs/figures/diagnostic_snapshot.svg", required)
        self.assertIn("docs/figures/pipeline_architecture.svg", required)
        self.assertIn("docs/figures/release_boundary.svg", required)
        self.assertIn("scripts/run_visual_pack.sh", required)
        self.assertIn("scripts/run_lgbm_benchmark.sh", required)
        self.assertIn("scripts/run_segment_set_model.sh", required)
        self.assertIn("src/segment_macro_betas/visual_pack.py", required)
        self.assertIn("src/segment_macro_betas/lgbm_benchmark.py", required)

    def test_release_audit_blocks_tracked_private_outputs(self) -> None:
        suffixes = release_audit.PRIVATE_DATA_LIKE_SUFFIXES
        for suffix in [".csv", ".html", ".png", ".parquet", ".pkl", ".xlsx"]:
            self.assertIn(suffix, suffixes)


if __name__ == "__main__":
    unittest.main()
