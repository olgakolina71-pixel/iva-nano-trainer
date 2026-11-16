import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from iva_nano.model import IVANano


def _ban_repeated_ngrams(logits, token_ids, ngram_size):
    """Prevent a continuation from entering short token-level repetition loops."""
    if ngram_size <= 0 or len(token_ids) + 1 < ngram_size:
        return logits
    prefix = tuple(token_ids[-(ngram_size - 1):]) if ngram_size > 1 else ()
    banned = []
    for index in range(len(token_ids) - ngram_size + 1):
        ngram = token_ids[index:index + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            banned.append(ngram[-1])
    if banned:
        logits = logits.clone()
        logits[list(set(banned))] = float("-inf")
    return logits


def sample_next_token(
    logits, history_ids, temperature=0.8, top_k=40, top_p=0.9,
    repetition_penalty=1.12, repetition_window=128, no_repeat_ngram_size=3,
):
    """Sample with nucleus filtering and safeguards against degenerate repetition."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    scores = logits.float().clone()
    recent = list(dict.fromkeys(history_ids[-repetition_window:]))
    if repetition_penalty > 1 and recent:
        ids = torch.tensor(recent, dtype=torch.long, device=scores.device)
        selected = scores[ids]
        scores[ids] = torch.where(selected < 0, selected * repetition_penalty, selected / repetition_penalty)
    scores = _ban_repeated_ngrams(scores, history_ids, no_repeat_ngram_size)
    if not torch.isfinite(scores).any():
        scores = logits.float().clone()
    scores = scores / temperature

    k = min(top_k, scores.numel())
    if k < scores.numel():
        threshold = torch.topk(scores, k).values[-1]
        scores[scores < threshold] = float("-inf")

    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[sorted_indices[remove]] = float("-inf")

    probabilities = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        return torch.argmax(logits)
    return torch.multinomial(probabilities, num_samples=1).squeeze(0)


@torch.no_grad()
def generate(
    model, tokenizer, prompt, device, max_new_tokens=80, temperature=0.8,
    top_k=40, top_p=0.9, repetition_penalty=1.12,
):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    ids = tokenizer.encode(prompt, out_type=int)
    if not ids:
        ids = [tokenizer.bos_id()]
    context = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)
    generated = []
    model.eval()
    for _ in range(max_new_tokens):
        window = context[:, -model.context_length:]
        logits, _ = model(window)
        next_id = sample_next_token(
            logits[0, -1, :], ids + generated,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        token_id = int(next_id.item())
        if token_id == tokenizer.eos_id():
            break
        generated.append(token_id)
        context = torch.cat([context, next_id.view(1, 1)], dim=1)
        partial = tokenizer.decode(generated)
        if "\n### Пользователь:" in partial or "\n### IVA:" in partial:
            break
    result = tokenizer.decode(generated).strip()
    for marker in ("\n### Пользователь:", "\n### IVA:"):
        result = result.split(marker, 1)[0].strip()
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained IVA checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/artifacts/iva_nano_40m_checkpoint.pt")
    parser.add_argument("--tokenizer", default="checkpoints/iva_tokenizer.model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu or cuda")
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
    print(generate(
        model, tokenizer, args.prompt, device, args.max_new_tokens,
        args.temperature, args.top_k, args.top_p, args.repetition_penalty,
    ))


if __name__ == "__main__":
    main()
