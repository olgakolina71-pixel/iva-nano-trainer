import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from iva_nano.data import TokenBlockDataset, load_approved_texts, train_tokenizer, validate_reviewed_training_manifest
from iva_nano.memmap_data import MemmapTokenBlockDataset
from iva_nano.model import IVANano
from iva_nano.storage import LocalArtifactStore


def resolve_device(requested: str) -> torch.device:
    """Resolve cpu/cuda/tpu, including registered PrivateUse1 TPU backends."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            backend_name = torch._C._get_privateuse1_backend_name()
        except (AttributeError, RuntimeError):
            backend_name = ""
        if backend_name == "tpu":
            return torch.device("tpu")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "tpu":
        try:
            backend_name = torch._C._get_privateuse1_backend_name()
        except (AttributeError, RuntimeError):
            backend_name = ""
        if backend_name != "tpu":
            raise RuntimeError(
                "TPU was requested, but no 'tpu' PrivateUse1 backend is registered. "
                "Install and initialize the TPU runtime before starting training."
            )
    if requested not in {"cpu", "cuda", "tpu"}:
        raise ValueError("training.device must be auto, cpu, cuda, or tpu")
    return torch.device(requested)


def _to_cpu(value):
    """Copy tensors in nested checkpoint structures to CPU for portable saves."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def learning_rate_at_step(step, cfg):
    """Linear warmup followed by cosine decay."""
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
def evaluate(model, loader, device, max_batches=100):
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


def save_checkpoint(store, name, model, optimizer, config, step, tokenizer_path):
    # Keep checkpoints portable: custom TPU device tensors are serialized on CPU.
    return store.save_checkpoint(name, {
        "model": _to_cpu(model.state_dict()),
        "optimizer": _to_cpu(optimizer.state_dict()),
        "model_config": config["model"],
        "training_config": config["training"],
        "model_id": config.get("model_id", "iva-nano"),
        "step": step,
        "parameter_count": model.parameter_count(),
        "tokenizer_path": str(tokenizer_path),
    }, torch)


def main():
    parser = argparse.ArgumentParser(description="Train or continue an IVA decoder-only language model.")
    parser.add_argument("--config", default="configs/iva_nano_40m.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--tokenizer-path", help="Reuse an existing SentencePiece model.")
    parser.add_argument("--token-ids", help="Optional uint16 disk-backed token cache created by scripts/prepare_training_cache.py.")
    parser.add_argument("--resume", help="Resume model weights and optimizer from a checkpoint.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Load model weights but start a fresh optimizer (useful for supervised fine-tuning).")
    parser.add_argument("--additional-steps", action="store_true", help="Treat training.max_steps as extra steps after the checkpoint step.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "tpu"), help="Override training.device from the config.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = config["training"]
    model_id = config.get("model_id", "iva-nano")
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device_name = args.device or train_cfg.get("device", "auto")
    device = resolve_device(device_name)

    if not args.smoke_test:
        validate_reviewed_training_manifest(args.manifest)
    texts = [] if args.token_ids else load_approved_texts(args.data_dir, args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import os
    artifact_root = Path(os.environ.get("IVA_ARTIFACT_DIR", str(out_dir / "artifacts")))
    artifact_store = LocalArtifactStore(artifact_root)

    tok_cfg = config["tokenizer"]
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else out_dir / (tok_cfg["model_prefix"] + ".model")
    if not tokenizer_path.exists():
        if args.tokenizer_path or args.token_ids:
            raise FileNotFoundError(f"Requested tokenizer does not exist: {tokenizer_path}")
        train_tokenizer(
            texts, str(out_dir / tok_cfg["model_prefix"]),
            int(tok_cfg["vocab_size"]), tok_cfg.get("model_type", "bpe"),
        )

    if args.token_ids:
        if train_cfg.get("objective", "pretraining") == "assistant_only":
            raise ValueError("Disk-backed token cache currently supports pretraining objective only")
        token_path = Path(args.token_ids)
        if not token_path.is_file():
            raise FileNotFoundError(f"Token cache does not exist: {token_path}")
        dataset = MemmapTokenBlockDataset(
            str(token_path), int(config["model"]["context_length"])
        )
    else:
        dataset = TokenBlockDataset(
            texts, str(tokenizer_path), int(config["model"]["context_length"]),
            assistant_only=train_cfg.get("objective", "pretraining") == "assistant_only",
        )
    if len(dataset) < 1:
        raise ValueError("Not enough tokenized text for one context block")

    token_count = int(dataset.token_count if args.token_ids else dataset.tokens.numel())
    minimum_tokens = int(train_cfg.get("minimum_training_tokens", 100_000_000))
    print(f"Model ID: {model_id}")
    print(f"Corpus tokens: {token_count:,}")
    if not args.smoke_test and token_count < minimum_tokens:
        raise ValueError(
            f"Corpus is too small for full training: {token_count:,} tokens; "
            f"minimum configured threshold is {minimum_tokens:,}. "
            "Add more reviewed, licensed and diverse text. Use --smoke-test only for pipeline checks."
        )

    eval_loader = None
    if len(dataset) >= 20:
        eval_size = max(1, int(len(dataset) * 0.05))
        train_size = len(dataset) - eval_size
        train_set, eval_set = random_split(
            dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(seed),
        )
        eval_loader = DataLoader(
            eval_set,
            batch_size=1 if args.smoke_test else int(train_cfg["batch_size"]),
            num_workers=0 if device.type == "tpu" else int(train_cfg.get("num_workers", 0)),
        )
    else:
        train_set = dataset
        print("Evaluation split skipped: fewer than 20 token blocks")

    batch_size = 1 if args.smoke_test else int(train_cfg["batch_size"])
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0 if device.type == "tpu" else int(train_cfg.get("num_workers", 0)),
    )
    if len(loader) == 0:
        raise ValueError("Training dataloader is empty")

    model = IVANano(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        # Load checkpoint tensors on CPU; load_state_dict transfers model/optimizer state.
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("model_config") != config["model"]:
            raise ValueError("Checkpoint model architecture does not match the selected config")
        model.load_state_dict(saved["model"])
        start_step = int(saved.get("step", 0))
        if not args.reset_optimizer and saved.get("optimizer"):
            optimizer.load_state_dict(saved["optimizer"])
        print(f"Resumed {saved.get('model_id', 'unknown model')} at step {start_step:,}")

    configured_steps = 1 if args.smoke_test else int(train_cfg["max_steps"])
    total_steps = start_step + configured_steps if args.additional_steps else configured_steps
    if total_steps <= start_step:
        raise ValueError(
            f"Checkpoint already reached step {start_step}; target is {total_steps}. "
            "Use --additional-steps to train for more steps."
        )
    schedule_cfg = dict(train_cfg)
    schedule_cfg["max_steps"] = total_steps

    print(f"Parameters: {model.parameter_count():,}")
    print(f"Device: {device}")
    print(f"Training blocks: {len(train_set):,}")
    print(f"Evaluation blocks: {len(eval_loader.dataset) if eval_loader else 0:,}")
    print(f"Target optimizer steps: {total_steps:,}")

    accumulation = 1 if args.smoke_test else max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    eval_interval = max(1, int(train_cfg.get("eval_interval", 100)))
    save_interval = max(1, int(train_cfg.get("save_interval", 250)))
    max_grad_norm = float(train_cfg["max_grad_norm"])
    checkpoint_name = train_cfg.get("checkpoint_name", "iva_nano_checkpoint.pt")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    metrics = {
        "model_id": model_id,
        "parameter_count": model.parameter_count(),
        "device": str(device),
        "start_step": start_step,
        "target_steps": total_steps,
        "corpus_tokens": token_count,
        "train_loss": [],
        "validation_loss": [],
    }

    while step < total_steps:
        for batch_index, (x, y) in enumerate(loader):
            window_start = (batch_index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            lr = learning_rate_at_step(step, schedule_cfg) if not args.smoke_test else float(train_cfg["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / window_size).backward()

            is_update = ((batch_index + 1) % accumulation == 0) or (batch_index + 1 == len(loader))
            if not is_update:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == start_step + 1 or step % 10 == 0:
                item = {"step": step, "loss": round(float(loss.item()), 5), "learning_rate": lr}
                metrics["train_loss"].append(item)
                print(json.dumps(item))

            if eval_loader and step % eval_interval == 0:
                val_loss = evaluate(model, eval_loader, device)
                item = {"step": step, "validation_loss": round(val_loss, 5)}
                metrics["validation_loss"].append(item)
                print(json.dumps(item))

            if step % save_interval == 0:
                save_checkpoint(
                    artifact_store, f"{model_id.replace('-', '_')}_step_{step}.pt",
                    model, optimizer, config, step, tokenizer_path,
                )
            if step >= total_steps:
                break

    final_validation = evaluate(model, eval_loader, device) if eval_loader else None
    if final_validation is not None:
        metrics["final_validation_loss"] = round(final_validation, 5)
        metrics["validation_loss"].append({"step": step, "validation_loss": round(final_validation, 5)})

    checkpoint = save_checkpoint(
        artifact_store, checkpoint_name, model, optimizer, config, step, tokenizer_path,
    )
    metrics["completed_steps"] = step
    metrics["checkpoint"] = str(checkpoint)
    metrics["final_train_loss"] = metrics["train_loss"][-1]["loss"] if metrics["train_loss"] else None
    metrics_path = artifact_root / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
