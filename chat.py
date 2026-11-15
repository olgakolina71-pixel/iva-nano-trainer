"""Interactive terminal chat for a trained IVA causal language model.

This is a local inference client. It does not make an untrained checkpoint conversational.
"""
import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from iva_nano.model import IVANano


def sample_next(logits, temperature, top_k):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    logits = logits / temperature
    k = min(max(1, top_k), logits.numel())
    values, indices = torch.topk(logits, k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, indices, values)
    return torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)


@torch.no_grad()
def answer(model, tokenizer, messages, system_prompt, device, max_new_tokens, temperature, top_k):
    transcript = [f"### Система:\n{system_prompt}"]
    for role, content in messages:
        header = "Пользователь" if role == "user" else "IVA"
        transcript.append(f"### {header}:\n{content}")
    prompt = "\n".join(transcript) + "\n### IVA:\n"
    ids = tokenizer.encode(prompt, out_type=int)
    if not ids:
        ids = [tokenizer.bos_id()]
    ids = ids[-model.context_length:]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    model.eval()
    for _ in range(max_new_tokens):
        window = context[:, -model.context_length:]
        logits, _ = model(window)
        next_id = sample_next(logits[0, -1, :], temperature, top_k)
        token_id = int(next_id.item())
        if token_id == tokenizer.eos_id():
            break
        generated.append(token_id)
        context = torch.cat([context, next_id.view(1, 1)], dim=1)
        text = tokenizer.decode(generated)
        if "\n### Пользователь:" in text or "\n### IVA:" in text:
            break
    result = tokenizer.decode(generated).strip()
    for marker in ("\n### Пользователь:", "\n### IVA:"):
        result = result.split(marker, 1)[0].strip()
    return result


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained IVA model.")
    parser.add_argument("--checkpoint", default="checkpoints/artifacts/iva_nano_40m_checkpoint.pt")
    parser.add_argument("--tokenizer", default="checkpoints/iva_tokenizer.model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--system-prompt",
        default="Ты IVA — Интеллектуальный Вычислительный Ассистент. Отвечай по-русски, ясно и честно. Если не знаешь ответа, прямо скажи об этом.",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = IVANano(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    print(f"IVA model: {checkpoint.get('model_id', 'unknown')}")
    print(f"Parameters: {checkpoint.get('parameter_count', model.parameter_count()):,}")
    print(f"Device: {device}")
    print("Введите /reset для очистки истории, /exit для выхода.")
    history = []
    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо встречи!")
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
        )
        print(f"IVA: {response or '[модель не сгенерировала ответ]'}")
        history.append(("assistant", response))


if __name__ == "__main__":
    main()
