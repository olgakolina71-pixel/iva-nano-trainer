import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from iva_nano.model import IVANano


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new_tokens=80, temperature=0.8, top_k=40):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    ids = tokenizer.encode(prompt, out_type=int)
    if not ids:
        ids = [tokenizer.bos_id()]
    context = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new_tokens):
        context = context[:, -model.context_length:]
        logits, _ = model(context)
        next_logits = logits[0, -1, :] / temperature
        k = min(top_k, next_logits.numel())
        values, indices = torch.topk(next_logits, k)
        filtered = torch.full_like(next_logits, float("-inf"))
        filtered.scatter_(0, indices, values)
        probs = torch.softmax(filtered, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_id.view(1, 1)], dim=1)
        if int(next_id.item()) == tokenizer.eos_id():
            break
    return tokenizer.decode(context[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained IVA Nano checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/artifacts/iva_nano_40m_checkpoint.pt")
    parser.add_argument("--tokenizer", default="checkpoints/iva_tokenizer.model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train the model first.")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}. Train the tokenizer first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = IVANano(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    result = generate(
        model, tokenizer, args.prompt, device,
        args.max_new_tokens, args.temperature, args.top_k,
    )
    print(result)


if __name__ == "__main__":
    main()
