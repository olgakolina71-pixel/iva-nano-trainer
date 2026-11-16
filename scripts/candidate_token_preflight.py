#!/usr/bin/env python3
"""Estimate SentencePiece token volume for a quarantined candidate corpus.

This utility is intentionally read-only with respect to source data and approval:
it never edits manifests, grants rights, or starts model training.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import sentencepiece as spm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 candidate text file")
    parser.add_argument("--output", default="reports/candidate-token-preflight.json")
    parser.add_argument("--tokenizer-prefix", default="/tmp/iva-candidate-tokenizer")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--sample-size", type=int, default=300000)
    parser.add_argument("--batch-records", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-tokens", type=int, default=100_000_000)
    parser.add_argument("--target-tokens", type=int, default=800_000_000)
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(source)
    if min(args.vocab_size, args.sample_size, args.batch_records, args.threads) < 1:
        raise ValueError("Tokenizer and batching settings must be positive")

    started = time.time()
    prefix = Path(args.tokenizer_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Train the candidate tokenizer on a shuffled sample, not the entire 847 MB
    # file. The same tokenizer settings should be reused if this candidate is
    # later approved; this sample-based count is a readiness estimate.
    spm.SentencePieceTrainer.train(
        input=str(source),
        model_prefix=str(prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
        input_sentence_size=args.sample_size,
        shuffle_input_sentence=True,
        max_sentence_length=65536,
        num_threads=args.threads,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")

    token_count = 0
    record_count = 0
    character_count = 0
    replacement_characters = 0
    null_characters = 0
    batch = []

    def flush_batch():
        nonlocal token_count, batch
        if batch:
            # Chunking bounds RAM. Newline whitespace is retained between
            # records; only chunk boundaries add a negligible tokenizer prefix.
            token_count += len(processor.encode("\n".join(batch), out_type=int))
            batch = []

    with source.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            record_count += 1
            character_count += len(line)
            replacement_characters += line.count("\ufffd")
            null_characters += line.count("\x00")
            batch.append(line)
            if len(batch) >= args.batch_records:
                flush_batch()
    flush_batch()
    token_count += 1  # one EOS for this single source file, as in pretraining

    report = {
        "status": "candidate_token_estimate_only_not_approved",
        "source_file": source.name,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "tokenizer": {
            "type": "SentencePiece BPE",
            "vocab_size_requested": args.vocab_size,
            "sample_size": args.sample_size,
            "max_sentence_length": 65536,
            "batch_records": args.batch_records,
        },
        "records": record_count,
        "characters": character_count,
        "replacement_characters": replacement_characters,
        "null_characters": null_characters,
        "estimated_tokens": token_count,
        "minimum_tokens": args.minimum_tokens,
        "target_tokens": args.target_tokens,
        "meets_minimum_estimate": token_count >= args.minimum_tokens,
        "meets_target_estimate": token_count >= args.target_tokens,
        "tokenizer_model_sha256": sha256_file(Path(str(prefix) + ".model")),
        "elapsed_seconds": round(time.time() - started, 1),
        "rights_approved": False,
        "training_authorized": False,
        "limitations": [
            "This is a sample-tokenizer, chunked-count estimate, not the final training preflight.",
            "The source remains quarantined; this report does not review rights, privacy, or content quality.",
            "Chunk boundaries can add a small tokenization difference versus encoding the entire file at once.",
            "Do not set use=training or rights_status=reviewed_approved based on this report.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
