"""Evaluate a trained IVA checkpoint on held-out approved text and fixed prompts."""
import argparse
import json
import math
from pathlib import Path

import sentencepiece as spm
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from iva_nano.data import TokenBlockDataset, load_approved_texts
from iva_nano.model import IVANano
from generate import generate


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    losses = []
    token_count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
        token_count += int(y.numel())
    if not losses:
        raise ValueError("Evaluation split is empty")
    mean_loss = sum(losses) / len(losses)
    return {
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "batches": len(losses),
        "target_tokens": token_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate an IVA checkpoint on approved held-out text.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default="configs/iva_nano_40m.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="evaluation_report.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu or cuda")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_config") != config["model"]:
        raise ValueError("Checkpoint architecture does not match evaluation config")
    if checkpoint.get("parameter_count") != sum(p.numel() for p in IVANano(config["model"]).parameters() if p.requires_grad):
        raise ValueError("Checkpoint parameter-count metadata does not match model architecture")

    texts = load_approved_texts(args.data_dir, args.manifest)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    dataset = TokenBlockDataset(texts, args.tokenizer, int(config["model"]["context_length"]))
    if len(dataset) < 2:
        raise ValueError("At least two token blocks are required for held-out evaluation")
    eval_size = max(1, int(len(dataset) * 0.05))
    _, eval_set = random_split(dataset, [len(dataset) - eval_size, eval_size],
                               generator=torch.Generator().manual_seed(2026))
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

    model = IVANano(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    scores = evaluate_loss(model, loader, device)

    prompts = [
        "### Пользователь:\nПривет!\n### IVA:\n",
        "Россия — это",
        "Объясни простыми словами, что такое нейронная сеть.",
    ]
    samples = []
    for prompt in prompts:
        generated = generate(model, tokenizer, prompt, device, max_new_tokens=80, temperature=0.8, top_k=40)
        samples.append({"prompt": prompt, "completion": generated})

    report = {
        "model_id": checkpoint.get("model_id", "unknown"),
        "parameter_count": checkpoint.get("parameter_count"),
        "checkpoint_step": checkpoint.get("step"),
        "device": device,
        "evaluation": scores,
        "warning": "Perplexity is a corpus-fit metric, not proof of instruction-following or factual accuracy. Review generated samples manually.",
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
