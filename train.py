import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from iva_nano.data import TokenBlockDataset, load_approved_texts, train_tokenizer
from iva_nano.model import IVANano


def main():
    parser = argparse.ArgumentParser(description="Train IVA Nano on licensed local text.")
    parser.add_argument("--config", default="configs/iva_nano_40m.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = config["training"]
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device_name = train_cfg.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    texts = load_approved_texts(args.data_dir, args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_cfg = config["tokenizer"]
    tokenizer_path = out_dir / (tok_cfg["model_prefix"] + ".model")
    if not tokenizer_path.exists():
        train_tokenizer(
            texts, str(out_dir / tok_cfg["model_prefix"]),
            int(tok_cfg["vocab_size"]), tok_cfg.get("model_type", "bpe"),
        )

    dataset = TokenBlockDataset(texts, str(tokenizer_path), int(config["model"]["context_length"]))
    if len(dataset) == 0:
        raise ValueError("Not enough tokenized text for one context block")
    loader = DataLoader(
        dataset, batch_size=1 if args.smoke_test else int(train_cfg["batch_size"]),
        shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = IVANano(config["model"]).to(device)
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Device: {device}")
    print(f"Training blocks: {len(dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    max_steps = 1 if args.smoke_test else int(train_cfg["max_steps"])
    model.train()
    step = 0
    while step < max_steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                print(json.dumps({"step": step, "loss": round(float(loss.item()), 5)}))
            if step >= max_steps:
                break

    checkpoint = out_dir / "iva_nano_checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "model_config": config["model"],
        "step": step,
        "parameter_count": model.parameter_count(),
    }, checkpoint)
    print(f"Saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
