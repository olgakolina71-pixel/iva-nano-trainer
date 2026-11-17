"""Run one real IVA Nano Transformer training step on a registered TPU backend.

Usage:
  python scripts/tpu_smoke_test.py --config configs/iva_nano_40m.yaml

This checks the actual model (including causal attention), loss, backward pass,
AdamW update, and a CPU-portable checkpoint round trip. It does not train on
the full corpus or modify existing checkpoints.
"""
import argparse
import tempfile
from pathlib import Path

import torch
import yaml

from iva_nano.model import IVANano
from train import resolve_device, _to_cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iva_nano_40m.yaml")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device("tpu")
    model = IVANano(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    vocab_size = int(config["model"]["vocab_size"])
    x = torch.randint(0, vocab_size, (args.batch_size, args.sequence_length), device=device)
    y = torch.randint(0, vocab_size, (args.batch_size, args.sequence_length), device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if logits.device.type != "tpu":
        raise RuntimeError(f"Expected TPU logits, got {logits.device}")
    print(f"Forward pass: OK ({logits.device})")
    print(f"Backward pass: OK")
    print(f"AdamW update: OK")
    print(f"Loss: {loss.item():.6f}")

    portable = {
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "model_config": config["model"],
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "iva_tpu_smoke.pt"
        torch.save(portable, path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
        check_model = IVANano(config["model"])
        check_model.load_state_dict(restored["model"])
        print(f"CPU checkpoint save/load: OK ({path.stat().st_size:,} bytes)")

    print("TPU smoke test PASSED. This is not yet a speed benchmark.")


if __name__ == "__main__":
    main()
