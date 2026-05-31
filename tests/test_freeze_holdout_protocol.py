import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.freeze_holdout_protocol import freeze_protocol


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class FreezeHoldoutProtocolTests(unittest.TestCase):
    def test_freeze_protocol_selects_development_candidate_without_opening_holdout(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "runs/panel/manifests/monthly_panel.json",
                {"status": "ok", "checks": {"max_date": "2025-12-31", "activation_rule_violations": 0}},
            )
            write_json(
                root / "runs/lgbm/manifests/lgbm_benchmark.json",
                {
                    "status": "ok",
                    "checks": {"best_spread_variant": "all_plus_macro"},
                    "outputs": {"variants": {"all_plus_macro": {"status": "ok"}}},
                },
            )
            write_json(root / "runs/factor/manifests/factor_robustness.json", {"status": "ok"})
            write_json(root / "runs/claim/manifests/claim_ledger.json", {"status": "ok", "checks": {"validation_failures": 0}})
            write_json(root / "runs/pub/manifests/publication_tables.json", {"status": "ok", "checks": {"review_failures": 0}})
            write_json(root / "runs/visual/manifests/visual_pack.json", {"status": "ok"})
            (root / "artifacts/tables/lgbm").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "variant": "all_plus_macro",
                        "prediction_rows": 10,
                        "mean_rank_ic": 0.02,
                        "t_rank_ic": 2.0,
                        "mean_q5_minus_q1": 0.01,
                        "t_q5_minus_q1": 3.0,
                    }
                ]
            ).to_csv(root / "artifacts/tables/lgbm/lgbm_summary.csv", index=False)
            (root / "artifacts/tables/factor").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "model_family": "lgbm",
                        "variant": "all_plus_macro",
                        "mean_net_q5_minus_q1": 0.009,
                        "t_net_q5_minus_q1": 2.7,
                        "gross_alpha": 0.008,
                        "gross_alpha_t": 2.4,
                        "alpha_months": 81,
                    }
                ]
            ).to_csv(root / "artifacts/tables/factor/factor_robustness_summary.csv", index=False)

            manifest = freeze_protocol(
                root,
                panel_run_id="panel",
                lgbm_run_id="lgbm",
                factor_run_id="factor",
                claim_run_id="claim",
                publication_run_id="pub",
                visual_run_id="visual",
                model_family="lgbm",
                variant="all_plus_macro",
            )

        self.assertEqual(manifest["status"], "frozen")
        self.assertFalse(manifest["holdout_opened"])
        self.assertEqual(manifest["selected_model"]["variant"], "all_plus_macro")
        self.assertEqual(manifest["checks"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
