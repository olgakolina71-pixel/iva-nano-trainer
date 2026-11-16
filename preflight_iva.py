"""Readiness audit for a real IVA Nano pretraining run.

This script never changes dataset approval or starts training. It reports exactly
which reviewed corpus is available and whether it passes the configured token gate.
"""
import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm
import yaml

from iva_nano.data import load_approved_texts, train_tokenizer


def count_model_parameters(model_cfg):
    d = int(model_cfg["d_model"])
    vocab = int(model_cfg["vocab_size"])
    context = int(model_cfg["context_length"])
    layers = int(model_cfg["n_layers"])
    ratio = int(model_cfg["mlp_ratio"])
    total = vocab * d + context * d + layers * (4 * d * d + 2 * ratio * d * d + (ratio + 5) * d) + 2 * d
    if not model_cfg.get("tie_embeddings", True):
        total += vocab * d
    return total


def inspect_manifest(manifest_path):
    path = Path(manifest_path)
    if not path.exists():
        return [], ["Dataset manifest is missing."]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"Dataset manifest cannot be read: {exc}"]
    entries = manifest.get("datasets", [])
    approved = [item for item in entries if item.get("use") == "training"]
    pending = [
        {"id": item.get("id", "<unknown>"), "use": item.get("use", "unspecified"),
         "rights_status": item.get("rights_status", "unspecified")}
        for item in entries if item.get("use") != "training"
    ]
    return pending, ([] if approved else ["No dataset is explicitly approved with use='training'."])


def audit(config_path, data_dir, manifest_path, tokenizer_path=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_cfg = config["model"]
    train_cfg = config["training"]
    parameter_count = count_model_parameters(model_cfg)
    pending, blockers = inspect_manifest(manifest_path)

    try:
        texts = load_approved_texts(str(data_dir), str(manifest_path))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        texts = []
        blockers.append(str(exc))

    report = {
        "model_id": config.get("model_id", "iva-nano"),
        "parameter_count": parameter_count,
        "context_length": int(model_cfg["context_length"]),
        "vocab_size": int(model_cfg["vocab_size"]),
        "minimum_training_tokens": int(train_cfg.get("minimum_training_tokens", 100_000_000)),
        "target_training_tokens": int(train_cfg.get("target_training_tokens", 800_000_000)),
        "approved_sources_pending_review": pending,
        "source_count": len(texts),
        "characters": sum(len(text) for text in texts),
        "token_count": 0,
        "usable_context_blocks": 0,
        "blockers": list(dict.fromkeys(blockers)),
        "status": "blocked",
    }

    if texts:
        tokenizer_file = Path(tokenizer_path) if tokenizer_path else None
        if tokenizer_file and tokenizer_file.exists():
            processor = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
        else:
            with tempfile.TemporaryDirectory(prefix="iva-preflight-") as temp_dir:
                prefix = Path(temp_dir) / "iva_preflight"
                model_file = train_tokenizer(
                    texts, str(prefix), int(config["tokenizer"]["vocab_size"]),
                    config["tokenizer"].get("model_type", "bpe"),
                )
                processor = spm.SentencePieceProcessor(model_file=model_file)
                report["token_count"] = sum(
                    len(processor.encode(text, out_type=int)) + 1 for text in texts
                )
        if tokenizer_file and tokenizer_file.exists():
            report["token_count"] = sum(
                len(processor.encode(text, out_type=int)) + 1 for text in texts
            )
        report["usable_context_blocks"] = max(
            0, (report["token_count"] - 1) // int(model_cfg["context_length"])
        )

    if report["token_count"] >= report["minimum_training_tokens"]:
        report["status"] = "ready_for_training_review"
        report["blockers"] = [
            item for item in report["blockers"]
            if "No dataset is explicitly approved" not in item
        ]
    elif texts:
        report["blockers"].append(
            f"Approved corpus has {report['token_count']:,} tokens; "
            f"configured minimum is {report['minimum_training_tokens']:,}."
        )

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit IVA Nano corpus readiness without training.")
    parser.add_argument("--config", default="configs/iva_nano_40m.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--tokenizer", help="Optional existing matching SentencePiece tokenizer.")
    parser.add_argument("--output", default="reports/iva-nano-preflight.json")
    args = parser.parse_args()

    report = audit(args.config, args.data_dir, args.manifest, args.tokenizer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
