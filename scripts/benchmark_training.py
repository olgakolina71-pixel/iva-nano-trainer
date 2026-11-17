"""Benchmark IVA Nano training-step throughput on the selected device.

Example:
  python scripts/benchmark_training.py --device tpu --batch-sizes 1,2,4 --sequence-lengths 256,512,1024

This uses synthetic tokens and reports optimizer steps/second and tokens/second.
It does not read the corpus or modify training checkpoints.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from iva_nano.model import IVANano
from train import resolve_device


def benchmark_case(config, device, batch_size, sequence_length, warmup, steps):
    model_config = dict(config["model"])
    model_config["context_length"] = max(
        int(model_config["context_length"]), sequence_length
    )
    model = IVANano(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training", {}).get("learning_rate", 1e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 0.1)),
    )
    vocab_size = int(model_config["vocab_size"])
    x = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss

    model.train()
    last_loss = None
    for _ in range(warmup):
        last_loss = train_step()
    if last_loss is not None:
        # Force completion of queued device work before starting the timer.
        float(last_loss.detach().cpu())

    started = time.perf_counter()
    for _ in range(steps):
        last_loss = train_step()
    # The device-to-host read synchronizes pending work before elapsed time is read.
    final_loss = float(last_loss.detach().cpu())
    elapsed = time.perf_counter() - started

    result = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "optimizer_steps": steps,
        "seconds": round(elapsed, 3),
        "steps_per_second": round(steps / elapsed, 3),
        "tokens_per_second": round(steps * batch_size * sequence_length / elapsed),
        "final_synthetic_loss": round(final_loss, 5),
        "device": str(device),
    }
    del optimizer, model, x, y, last_loss
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Measure IVA Nano training throughput with synthetic data."
    )
    parser.add_argument("--config", default="configs/iva_nano_40m.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "tpu"), default="auto")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--sequence-lengths", default="256,512,1024")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    if args.warmup < 0 or args.steps < 1:
        raise ValueError("--warmup must be >= 0 and --steps must be >= 1")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    sequence_lengths = [
        int(value) for value in args.sequence_lengths.split(",") if value.strip()
    ]
    if not batch_sizes or not sequence_lengths:
        raise ValueError("Provide at least one batch size and sequence length")

    print(json.dumps({
        "event": "benchmark_start",
        "device": str(device),
        "warmup": args.warmup,
        "timed_steps": args.steps,
    }))

    results = []
    for sequence_length in sequence_lengths:
        for batch_size in batch_sizes:
            if batch_size < 1 or sequence_length < 2:
                print(json.dumps({
                    "status": "skipped",
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "reason": "batch size must be >= 1 and sequence length >= 2",
                }))
                continue
            try:
                result = benchmark_case(
                    config, device, batch_size, sequence_length,
                    args.warmup, args.steps,
                )
                result["status"] = "ok"
                results.append(result)
                print(json.dumps(result))
            except (RuntimeError, MemoryError) as error:
                # One oversized test must not prevent smaller cases from running.
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(json.dumps({
                    "status": "failed",
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "error": str(error)[:500],
                }))

    print(json.dumps({"event": "benchmark_complete", "results": results}))


if __name__ == "__main__":
    main()
