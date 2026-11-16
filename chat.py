"""Interactive multi-turn chat for a trained IVA causal language model.

This client improves prompt-window management and sampling. It cannot make an
untrained or undertrained checkpoint into a capable assistant.
"""
import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from iva_nano.model import IVANano
from generate import sample_next_token


def build_chat_prompt(tokenizer, messages, system_prompt, context_length, reserve_tokens):
    """Keep the system instruction and newest turns inside the model context."""
    retained = list(messages)
    budget = max(8, context_length - reserve_tokens)
    while True:
        sections = [f"### Система:\\n{system_prompt}"]
        for role, content in retained:
            header = "Пользователь" if role == "user" else "IVA"
            sections.append(f"### {header}:\\n{content}")
        prompt = "\\n".join(sections) + "\\n### IVA:\\n"
        ids = tokenizer.encode(prompt, out_type=int)
        if len(ids) <= budget or len(retained) <= 1:
            return ids[-budget:]
        # Drop the oldest complete message; the current user turn is retained.
        retained.pop(0)


@torch.no_grad()
def answer(model, tokenizer, messages, system_prompt, device, max_new_tokens,
           temperature, top_k, top_p=0.9, repetition_penalty=1.12):
    ids = build_chat_prompt(
        tokenizer, messages, system_prompt, model.context_length, max_new_tokens
    )
    if not ids:
        ids = [tokenizer.bos_id()]
    context = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
    generated = []
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(context[:, -model.context_length:])
        next_id = sample_next_token(
            logits[0, -1, :], ids + generated, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        )
        token_id = int(next_id.item())
        if token_id == tokenizer.eos_id():
            break
        generated.append(token_id)
        context = torch.cat([context, next_id.view(1, 1)], dim=1)
        partial = tokenizer.decode(generated)
        if "\\n### Пользователь:" in partial or "\\n### IVA:" in partial:
            break
    result = tokenizer.decode(generated).strip()
    for marker in ("\\n### Пользователь:", "\\n### IVA:"):
        result = result.split(marker, 1)[0].strip()
    return result


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained IVA model.")
    parser.add_argument("--checkpoint", default="checkpoints/artifacts/iva_nano_40m_checkpoint.pt")
    parser.add_argument("--tokenizer", default="checkpoints/iva_tokenizer.model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument(
        "--system-prompt",
        default="Ты IVA — Интеллектуальный Вычислительный Ассистент. Отвечай по-русски, "
                "ясно и по существу. Выполняй просьбу пользователя. Не выдумывай факты: "
                "если не знаешь, так и скажи. Учитывай предыдущие сообщения диалога.",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1 or args.temperature <= 0 or args.top_k < 1:
        parser.error("max-new-tokens, temperature and top-k must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device not in {"cpu", "cuda"}:
        parser.error("--device must be auto, cpu or cuda")

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = IVANano(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    print(f"IVA model: {checkpoint.get('model_id', 'unknown')}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Device: {device}")
    print("Введите /reset для очистки истории, /exit для выхода.")
    history = []
    while True:
        try:
            user_text = input("\\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\\nДо встречи!")
            break
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("История очищена.")
            continue
        history.append(("user", user_text))
        response = answer(
            model, tokenizer, history, args.system_prompt, device,
            args.max_new_tokens, args.temperature, args.top_k,
            args.top_p, args.repetition_penalty,
        )
        print(f"IVA: {response or '[модель не сгенерировала ответ]'}")
        history.append(("assistant", response))


if __name__ == "__main__":
    main()
