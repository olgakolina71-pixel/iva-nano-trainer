import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from iva_nano.storage import LocalArtifactStore


class StorageTests(unittest.TestCase):
    def test_checkpoint_is_saved_and_manifested(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalArtifactStore(tmp)
            path = store.save_checkpoint("test.pt", {"step": 3}, torch)
            self.assertTrue(path.exists())
            self.assertEqual(torch.load(path, weights_only=True)["step"], 3)
            manifest = json.loads((Path(tmp) / "artifact_manifest.json").read_text(encoding="utf-8"))
            item = manifest["artifacts"][0]
            self.assertEqual(item["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(item["storage"], "local_filesystem")


if __name__ == "__main__":
    unittest.main()
