import json
import tempfile
import unittest
from pathlib import Path

from prepare_sft import prepare


class PrepareSFTTests(unittest.TestCase):
    def test_refuses_unreviewed_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "data" / "raw" / "hf"
            source_root.mkdir(parents=True)
            source = source_root / "dialogues.txt"
            source.write_text("### Пользователь:\\nПривет\\n### IVA:\\nЗдравствуйте\\n", encoding="utf-8")
            manifest = root / "hf_manifest.json"
            manifest.write_text(json.dumps({"datasets": [{
                "id": "dialogues",
                "path": str(source),
                "license": "CC0-1.0",
                "attribution": "Test source",
                "use": "pending_rights_review",
                "rights_status": "dataset_card_claim_unverified",
            }]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare(manifest, "dialogues", source_root, root / "sft")

    def test_copies_only_explicitly_reviewed_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "data" / "raw" / "hf"
            source_root.mkdir(parents=True)
            source = source_root / "dialogues.txt"
            source.write_text("### Пользователь:\\nПривет\\n### IVA:\\nЗдравствуйте\\n", encoding="utf-8")
            manifest = root / "hf_manifest.json"
            manifest.write_text(json.dumps({"datasets": [{
                "id": "dialogues",
                "path": str(source),
                "license": "CC0-1.0",
                "attribution": "Test source",
                "use": "training",
                "rights_status": "reviewed_approved",
            }]}), encoding="utf-8")
            result = prepare(manifest, "dialogues", source_root, root / "sft")
            self.assertTrue(Path(result["text"]).is_file())
            saved = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["datasets"][0]["use"], "training")
            self.assertEqual(saved["datasets"][0]["rights_status"], "reviewed_approved")


if __name__ == "__main__":
    unittest.main()
