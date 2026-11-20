"""Inference-only generalization check for the IVA Nano 10-dialogue LoRA adapter."""
import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
from peft import PeftModel

from iva_nano.model import IVANano
from generate import generate

# These prompts were not used in the ten-example memorization diagnostic.
PROMPTS = [
    "Привет, представься своими словами.",
    "Почему после дождя на улице появляются лужи?",
    "Что стоит взять на прогулку в холодную погоду?",
    "Зачем растениям нужен солнечный свет?",
    "Объясни простыми словами, что такое интернет.",
    "Чем кошка отличается от собаки?",
    "Назови две планеты Солнечной системы.",
    "Составь короткое поздравление с днём рождения.",
    "Переведи слово «дом» на английский язык.",
    "Продолжи последовательность: 2, 4, 6, 8, ...",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="generalization_report.json")
    args = parser.parse_args()

    device = "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = IVANano(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    results = []
    for prompt in PROMPTS:
        full_prompt = f"### Пользователь:\\n{prompt}\\n### IVA:\\n"
        answer = generate(
            model, tokenizer, full_prompt, device,
            max_new_tokens=48, temperature=1.0, top_k=1, top_p=1.0,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        item = {"prompt": prompt, "answer": answer}
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    Path(args.output).write_text(
        json.dumps({"mode": "inference_only", "prompts": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
