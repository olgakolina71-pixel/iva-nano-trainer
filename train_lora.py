"""Supervised LoRA fine-tuning for IVA Nano using PEFT."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from peft import LoraConfig, get_peft_model

from iva_nano.data import (
    TokenBlockDataset,
    load_approved_texts,
    validate_reviewed_training_manifest,
)
from iva_nano.hf_compat import audit_sentencepiece_tokenizer, load_safetensors_checkpoint
from iva_nano.experiment_tracking import start_experiment


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    if loader is None:
        return None
    model.eval()
    losses = []
    for index, (x, y) in enumerate(loader):
        if index >= max_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        if loss is not None and torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else None


def load_base_checkpoint(path, device):
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        state, model_config, metadata = load_safetensors_checkpoint(path, device=device)
        return state, model_config
    saved = torch.load(path, map_location=device, weights_only=False)
    if "model" not in saved or "model_config" not in saved:
        raise ValueError("Base checkpoint must contain model and model_config")
    return saved["model"], saved["model_config"]


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT for IVA Nano (PEFT).")
    parser.add_argument("--checkpoint", required=True, help="Reviewed IVA base checkpoint (.pt or .safetensors)")
    parser.add_argument("--tokenizer", default="checkpoints/iva_tokenizer.model")
    parser.add_argument("--config", default="configs/iva_nano_sft.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--output-dir", default="checkpoints/iva-nano-lora")
    parser.add_argument("--steps", type=int, help="Override configured optimizer steps")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    if args.rank < 1 or args.alpha < 1 or not 0 <= args.dropout < 1:
        parser.error("rank/alpha must be positive and dropout must be in [0, 1)")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = config["training"]
    tracker = start_experiment("iva-nano-lora-sft", config)
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device_name = train_cfg.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device not in {"cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")

    approved = validate_reviewed_training_manifest(args.manifest)
    texts = load_approved_texts(args.data_dir, args.manifest)
    tokenizer_report = audit_sentencepiece_tokenizer(args.tokenizer, int(config["model"]["vocab_size"]))

    from iva_nano.model import IVANano
    base_state, base_config = load_base_checkpoint(args.checkpoint, device)
    if base_config != config["model"]:
        raise ValueError("Base checkpoint architecture does not match the SFT config")
    base_model = IVANano(base_config).to(device)
    base_model.load_state_dict(base_state, strict=True)

    dataset = TokenBlockDataset(
        texts, args.tokenizer, int(base_config["context_length"]), assistant_only=True
    )
    supervised_tokens = int((dataset.labels != -100).sum().item())
    minimum_tokens = int(train_cfg.get("minimum_training_tokens", 200_000))
    if supervised_tokens < minimum_tokens:
        raise ValueError(
            f"Not enough reviewed assistant-response tokens: {supervised_tokens:,}; "
            f"configured minimum is {minimum_tokens:,}. Add reviewed dialogue examples."
        )
    if len(dataset) < 2:
        raise ValueError("Need at least two SFT blocks to create train/validation splits")

    eval_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - eval_size
    train_set, eval_set = random_split(
        dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(seed),
    )
    batch_size = int(train_cfg.get("batch_size", 2))
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size)
    if not len(loader):
        raise ValueError("SFT training dataloader is empty")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["qkv", "proj", "net.0", "net.2"],
        bias="none",
        task_type=None,
    )
    model = get_peft_model(base_model, lora_config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PEFT did not create any trainable LoRA parameters")
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    steps = int(args.steps if args.steps is not None else train_cfg.get("max_steps", 2000))
    if steps < 1:
        parser.error("--steps must be positive")
    accumulation = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    iterator = iter(loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    for step in range(1, steps + 1):
        losses = []
        for _ in range(accumulation):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                x, y = next(iterator)
            _, loss = model(x.to(device), y.to(device))
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Non-finite LoRA training loss")
            (loss / accumulation).backward()
            losses.append(float(loss.detach().item()))
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % 25 == 0 or step == steps:
            item = {"step": step, "train_loss": round(sum(losses) / len(losses), 6)}
            print(json.dumps(item))
            metrics.append(item)
            if tracker:
                tracker.log_metrics({"train/loss": item["train_loss"]}, step)

    validation_loss = evaluate(model, eval_loader, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    metadata = {
        "model_id": "iva-nano",
        "base_checkpoint": str(Path(args.checkpoint)),
        "base_checkpoint_sha256": sha256_file(args.checkpoint),
        "tokenizer_path": str(Path(args.tokenizer)),
        "tokenizer_sha256": tokenizer_report["sha256"],
        "model_config": base_config,
        "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
                 "target_modules": ["qkv", "proj", "net.0", "net.2"]},
        "optimizer_steps": steps,
        "supervised_tokens": supervised_tokens,
        "approved_dataset_ids": [entry.get("id") for entry in approved],
        "validation_loss": validation_loss,
        "training_metrics": metrics,
    }
    (output_dir / "iva_lora_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved IVA LoRA adapter: {output_dir}")
    if validation_loss is not None:
        print(f"Validation loss: {validation_loss:.6f}")
    if tracker:
        tracker.log_metrics({"validation/loss": validation_loss}, steps)
        tracker.finish()


if __name__ == "__main__":
    main()
