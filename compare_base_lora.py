"""Compare original IVA Nano checkpoint against its 10-dialogue LoRA adapter."""
import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
from peft import PeftModel

from iva_nano.model import IVANano
from generate import generate

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
    parser.add_argument("--output", default="base_vs_lora_report.json")
    args = parser.parse_args()

    device = "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model = IVANano(checkpoint["model_config"]).to(device)
    base_model.load_state_dict(checkpoint["model"], strict=True)
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    model.eval()

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    results = []
    for prompt in PROMPTS:
        full_prompt = f"### Пользователь:\n{prompt}\n### IVA:\n"
        with model.disable_adapter():
            base_answer = generate(
                model, tokenizer, full_prompt, device,
                max_new_tokens=48, temperature=1.0, top_k=1, top_p=1.0,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
            )
        lora_answer = generate(
            model, tokenizer, full_prompt, device,
            max_new_tokens=48, temperature=1.0, top_k=1, top_p=1.0,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        item = {
            "prompt": prompt,
            "base_answer": base_answer,
            "lora_answer": lora_answer,
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    report = {
        "mode": "inference_only_base_vs_lora",
        "training_performed": False,
        "prompts": results,
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
