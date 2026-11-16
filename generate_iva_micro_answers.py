"""Generate genuine answer samples from the existing IVA Micro smoke checkpoint.

This is deliberately NOT IVA Nano. The checkpoint was trained only for a tiny
CPU smoke test and is expected to produce weak, possibly incoherent text.
"""
import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch

from chat import answer
from iva_nano.model import IVANano

QUESTIONS = [
    ("micro-001", "Сколько будет 2 + 2? Объясни кратко."),
    ("micro-002", "Почему небо кажется голубым днём?"),
    ("micro-003", "Что такое нейронная сеть простыми словами?"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default="data/academy/iva_answers.jsonl")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("IVA Micro checkpoint or tokenizer is missing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = IVANano(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    records = []
    for record_id, question in QUESTIONS:
        generated = answer(
            model, tokenizer, [("user", question)],
            "Ты IVA Micro. Отвечай по-русски. Не продолжай диалог за пользователя.",
            device, max_new_tokens=48, temperature=0.7, top_k=20,
            top_p=0.9, repetition_penalty=1.15,
        ).strip()
        if not generated:
            print(f"[WARN] {record_id}: model returned an empty answer")
            continue
        records.append({
            "id": record_id,
            "question": question,
            "iva_answer": generated,
            "model_id": checkpoint.get("model_id", "iva-micro-smoke"),
            "checkpoint_type": "micro_diagnostic_not_nano",
        })
        print(f"[GENERATED] {record_id}: {generated[:180]}")

    if not records:
        raise RuntimeError("IVA Micro produced no non-empty answers; nothing will be fabricated")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} genuine IVA Micro generations to {output}")


if __name__ == "__main__":
    main()
