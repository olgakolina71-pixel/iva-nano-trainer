"""Tiny 10-example memorization diagnostic for IVA Nano.

This intentionally uses ten original synthetic examples. It is not a production
fine-tune and does not alter the base checkpoint.
"""
import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from iva_nano.model import IVANano


EXAMPLES = [
    ("Привет!", "Привет! Я IVA."),
    ("Как тебя зовут?", "Меня зовут IVA."),
    ("Кто ты?", "Я IVA, языковая модель проекта IVA Nano."),
    ("Сколько будет 2+2?", "2 + 2 = 4."),
    ("Какой цвет у неба в ясный день?", "Обычно голубой."),
    ("Назови столицу Франции.", "Столица Франции — Париж."),
    ("Сколько дней в неделе?", "В неделе семь дней."),
    ("Что делает кошка?", "Кошка может мяукать и мурлыкать."),
    ("На каком языке мы говорим?", "Мы говорим на русском языке."),
    ("Скажи слово МАЯК.", "МАЯК."),
]


class TinyDialogueDataset(Dataset):
    def __init__(self, tokenizer_path, context_length):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.items = []
        eos = self.sp.eos_id()
        for question, answer in EXAMPLES:
            prompt = f"### Пользователь:\n{question}\n### IVA:\n"
            prompt_ids = self.sp.encode(prompt, out_type=int)
            answer_ids = self.sp.encode(answer, out_type=int)
            ids = prompt_ids + answer_ids + [eos]
            labels = [-100] * len(prompt_ids) + answer_ids + [eos]
            if len(ids) - 1 > context_length:
                raise ValueError(f"Example exceeds context length: {question}")
            self.items.append((ids[:-1], labels[1:]))
        self.pad_id = self.sp.pad_id()
        self.context_length = context_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        ids, labels = self.items[index]
        pad = self.context_length - len(ids)
        return (
            torch.tensor(ids + [self.pad_id] * pad, dtype=torch.long),
            torch.tensor(labels + [-100] * pad, dtype=torch.long),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--output-dir", default="diagnostic-lora")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")
    random.seed(42)
    torch.manual_seed(42)
    device = args.device
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = saved["model_config"]
    model = IVANano(config).to(device)
    model.load_state_dict(saved["model"], strict=True)

    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    dataset = TinyDialogueDataset(args.tokenizer, min(160, int(config["context_length"])))
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["qkv", "proj", "net.0", "net.2"],
        bias="none", task_type=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=5e-4, weight_decay=0.0)

    model.train()
    losses = []
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(json.dumps({"step": step, "loss": round(losses[-1], 6)}), flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    from generate import generate
    model.eval()
    results = []
    for question, expected in EXAMPLES:
        prompt = f"### Пользователь:\n{question}\n### IVA:\n"
        actual = generate(
            model, tokenizer, prompt, device,
            max_new_tokens=32, temperature=1.0, top_k=1, top_p=1.0,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
        )
        item = {
            "question": question,
            "expected": expected,
            "actual": actual,
            "exact_match": actual.strip() == expected.strip(),
        }
        results.append(item)
        print("RESULT " + json.dumps(item, ensure_ascii=False), flush=True)

    report = {
        "examples": len(EXAMPLES),
        "steps": args.steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "exact_matches": sum(x["exact_match"] for x in results),
        "results": results,
    }
    (out / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SUMMARY " + json.dumps({
        "initial_loss": report["initial_loss"],
        "final_loss": report["final_loss"],
        "exact_matches": report["exact_matches"],
        "examples": report["examples"],
    }), flush=True)


if __name__ == "__main__":
    main()
