import json
import tempfile
import unittest
from pathlib import Path

from preflight_iva import audit, count_model_parameters, inspect_manifest


class PreflightTests(unittest.TestCase):
    def test_nano_parameter_target_is_exact(self):
        config = {
            "vocab_size": 24000, "context_length": 1024, "d_model": 528,
            "n_layers": 8, "mlp_ratio": 4, "tie_embeddings": True,
        }
        self.assertEqual(count_model_parameters(config), 40_015_008)

    def test_pending_sources_are_never_reported_as_approved(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps({
                "datasets": [
                    {"id": "candidate", "use": "pending_rights_review",
                     "rights_status": "dataset_card_claim_unverified"}
                ]
            }), encoding="utf-8")
            pending, blockers = inspect_manifest(manifest)
        self.assertEqual(pending[0]["id"], "candidate")
        self.assertTrue(any("No dataset is explicitly approved" in item for item in blockers))

    def test_parameter_count_respects_mlp_ratio(self):
        config = {
            "vocab_size": 100, "context_length": 16, "d_model": 32,
            "n_layers": 2, "n_heads": 4, "mlp_ratio": 2,
            "tie_embeddings": True,
        }
        expected = 100*32 + 16*32 + 2*(4*32*32 + 2*2*32*32 + 7*32) + 2*32
        self.assertEqual(count_model_parameters(config), expected)

    def test_unreviewed_training_entry_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.json"
            manifest.write_text(json.dumps({"datasets": [{
                "id": "candidate", "use": "training",
                "rights_status": "dataset_card_claim_unverified",
            }]}), encoding="utf-8")
            pending, blockers = inspect_manifest(manifest)
        self.assertTrue(any("No dataset is explicitly approved" in item for item in blockers))

    def test_missing_manifest_blocks_training_readiness(self):
        report = audit(
            "configs/iva_nano_40m.yaml",
            "data/raw",
            "data/manifest-does-not-exist.json",
        )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["token_count"], 0)
        self.assertTrue(report["blockers"])


if __name__ == "__main__":
    unittest.main()
