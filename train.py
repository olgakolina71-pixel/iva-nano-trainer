import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from iva_nano.data import TokenBlockDataset, load_approved_texts, train_tokenizer
from iva_nano.model import IVANano
from iva_nano.storage import LocalArtifactStore


def learning_rate_at_step(step, cfg):
    """Linear warmup followed by cosine decay to min_learning_rate."""
    base = float(cfg["learning_rate"])
    minimum = float(cfg.get("min_learning_rate", base))
    warmup = int(cfg.get("warmup_steps", 0))
    total = max(1, int(cfg["max_steps"]))
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    if total <= warmup:
        return minimum
    progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=20):
    if loader is None:
        return None
    model.eval()
    losses = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else None


def save_checkpoint(store, name, model, optimizer, config, step):
    return store.save_checkpoint(name, {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": config["model"],
        "training_config": config["training"],
        "step": step,
        "parameter_count": model.parameter_count(),
    }, torch)


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
    import os
    artifact_root = Path(os.environ.get("IVA_ARTIFACT_DIR", str(out_dir / "artifacts")))
    artifact_store = LocalArtifactStore(artifact_root)
    tok_cfg = config["tokenizer"]
    tokenizer_path = out_dir / (tok_cfg["model_prefix"] + ".model")
    if not tokenizer_path.exists():
        train_tokenizer(
            texts, str(out_dir / tok_cfg["model_prefix"]),
            int(tok_cfg["vocab_size"]), tok_cfg.get("model_type", "bpe"),
        )

    dataset = TokenBlockDataset(texts, str(tokenizer_path), int(config["model"]["context_length"]))
    if len(dataset) < 1:
        raise ValueError("Not enough tokenized text for one context block")

    token_count = int(dataset.tokens.numel())
    minimum_tokens = int(train_cfg.get("minimum_training_tokens", 1_000_000))
    print(f"Corpus tokens: {token_count:,}")
    if not args.smoke_test and token_count < minimum_tokens:
        raise ValueError(
            f"Corpus is too small for full training: {token_count:,} tokens; "
            f"minimum configured threshold is {minimum_tokens:,}. "
            "Add more reviewed, licensed text. Use --smoke-test only for pipeline checks."
        )

    eval_loader = None
    if len(dataset) >= 20:
        eval_size = max(1, int(len(dataset) * 0.05))
        train_size = len(dataset) - eval_size
        train_set, eval_set = random_split(
            dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(seed),
        )
        eval_loader = DataLoader(eval_set, batch_size=1 if args.smoke_test else int(train_cfg["batch_size"]))
    else:
        train_set = dataset
        print("Evaluation split skipped: fewer than 20 token blocks")

    batch_size = 1 if args.smoke_test else int(train_cfg["batch_size"])
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = IVANano(config["model"]).to(device)
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Device: {device}")
    print(f"Training blocks: {len(train_set)}")
    print(f"Evaluation blocks: {len(eval_loader.dataset) if eval_loader else 0}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    max_steps = 1 if args.smoke_test else int(train_cfg["max_steps"])
    accumulation = 1 if args.smoke_test else max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    eval_interval = max(1, int(train_cfg.get("eval_interval", 100)))
    save_interval = max(1, int(train_cfg.get("save_interval", 250)))
    max_grad_norm = float(train_cfg["max_grad_norm"])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    running_loss = 0.0
    metrics = {"model_id": config.get("model_id", "iva-micro"), "parameter_count": model.parameter_count(), "device": device, "max_steps": max_steps, "train_loss": [], "validation_loss": []}
    while step < max_steps:
        for x, y in loader:
            lr = learning_rate_at_step(step, train_cfg) if not args.smoke_test else float(train_cfg["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / accumulation).backward()
            running_loss += float(loss.item())
            micro_step += 1
            if micro_step % accumulation != 0 and step + 1 < max_steps:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or step % 10 == 0:
                item = {"step": step, "loss": round(running_loss / accumulation, 5), "learning_rate": lr}
                metrics["train_loss"].append(item)
                print(json.dumps(item))
            running_loss = 0.0

            if eval_loader and step % eval_interval == 0:
                val_loss = evaluate(model, eval_loader, device)
                item = {"step": step, "validation_loss": round(val_loss, 5)}
                metrics["validation_loss"].append(item)
                print(json.dumps(item))
            if step % save_interval == 0:
                save_checkpoint(artifact_store, f"iva_nano_step_{step}.pt", model, optimizer, config, step)
            if step >= max_steps:
                break

    checkpoint = save_checkpoint(artifact_store, "iva_nano_checkpoint.pt", model, optimizer, config, step)
    metrics["completed_steps"] = step
    metrics["checkpoint"] = str(checkpoint)
    metrics["final_train_loss"] = metrics["train_loss"][-1]["loss"] if metrics["train_loss"] else None
    metrics_path = artifact_root / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
