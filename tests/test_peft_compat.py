import unittest

import torch
from peft import LoraConfig, get_peft_model

from iva_nano.model import IVANano


class PeftCompatibilityTests(unittest.TestCase):
    def test_lora_injects_into_iva_linear_layers_and_freezes_base(self):
        config = {
            "vocab_size": 32, "context_length": 16, "d_model": 16,
            "n_layers": 2, "n_heads": 4, "mlp_ratio": 2,
            "dropout": 0.0, "tie_embeddings": True,
        }
        base = IVANano(config)
        peft_model = get_peft_model(
            base,
            LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0.0,
                target_modules=["qkv", "proj", "net.0", "net.2"],
                bias="none", task_type=None,
            ),
        )
        trainable = [name for name, parameter in peft_model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(any("lora_A" in name for name in trainable))
        self.assertTrue(all("lora_" in name for name in trainable))
        logits, loss = peft_model(torch.randint(0, 32, (2, 8)), torch.randint(0, 32, (2, 8)))
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
