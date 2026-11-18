import tempfile
import unittest
from pathlib import Path

import torch

from iva_nano.hf_compat import export_safetensors_checkpoint, load_safetensors_checkpoint
from iva_nano.model import IVANano


class SafetensorsCompatibilityTests(unittest.TestCase):
    def test_tied_embedding_checkpoint_round_trips(self):
        config = {
            "vocab_size": 32, "context_length": 16, "d_model": 16,
            "n_layers": 1, "n_heads": 4, "mlp_ratio": 2,
            "dropout": 0.0, "tie_embeddings": True,
        }
        model = IVANano(config)
        checkpoint = {
            "model": model.state_dict(),
            "model_config": config,
            "model_id": "iva-test",
            "step": 12,
            "parameter_count": model.parameter_count(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "iva.safetensors"
            export_safetensors_checkpoint(checkpoint, path)
            state, loaded_config, metadata = load_safetensors_checkpoint(path)
            restored = IVANano(loaded_config)
            restored.load_state_dict(state, strict=True)
            self.assertEqual(metadata["step"], "12")
            self.assertTrue(torch.equal(
                model.token_embedding.weight, restored.token_embedding.weight
            ))
            self.assertTrue(torch.equal(
                restored.token_embedding.weight, restored.lm_head.weight
            ))


if __name__ == "__main__":
    unittest.main()
