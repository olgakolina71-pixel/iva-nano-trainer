import argparse
import json
import sys
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
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than zero")
    if repetition_window < 1 or no_repeat_ngram_size < 0:
        raise ValueError("invalid repetition controls")

    scores = logits.float().clone()
    recent = list(dict.fromkeys(history_ids[-repetition_window:]))
    if repetition_penalty > 1 and recent:
        ids = torch.tensor(recent, dtype=torch.long, device=scores.device)
        selected = scores[ids]
        scores[ids] = torch.where(
            selected < 0, selected * repetition_penalty,
            selected / repetition_penalty,
        )
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
    repetition_window=128, no_repeat_ngram_size=3, stats=None,
):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    eos_id = tokenizer.eos_id()
    if eos_id is None or eos_id < 0:
        raise ValueError("Tokenizer has no valid EOS token ID")

    ids = tokenizer.encode(prompt, out_type=int)
    if not ids:
        bos_id = tokenizer.bos_id()
        if bos_id is None or bos_id < 0:
            raise ValueError("Tokenizer has no valid BOS token ID for an empty prompt")
        ids = [bos_id]

    context = torch.tensor(
        [ids[-model.context_length:]], dtype=torch.long, device=device
    )
    generated = []
    stop_reason = "max_new_tokens"
    model.eval()

    for _ in range(max_new_tokens):
        window = context[:, -model.context_length:]
        logits, _ = model(window)
        next_id = sample_next_token(
            logits[0, -1, :], ids + generated,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        token_id = int(next_id.item())
        if token_id == eos_id:
            stop_reason = "eos"
            break

        generated.append(token_id)
        context = torch.cat([context, next_id.view(1, 1)], dim=1)
        partial = tokenizer.decode(generated)
        if "\n### Пользователь:" in partial or "\n### IVA:" in partial:
            stop_reason = "chat_marker"
            break

    result = tokenizer.decode(generated).strip()
    for marker in ("\n### Пользователь:", "\n### IVA:"):
        result = result.split(marker, 1)[0].strip()

    if stats is not None:
        stats.update({
            "eos_id": int(eos_id),
            "generated_tokens": len(generated),
            "stop_reason": stop_reason,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "repetition_window": int(repetition_window),
            "no_repeat_ngram_size": int(no_repeat_ngram_size),
        })
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
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--report-stats", action="store_true", help="Print generation diagnostics to stderr")
    parser.add_argument("--lora-adapter", help="Optional PEFT LoRA adapter directory")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.repetition_window < 1 or args.no_repeat_ngram_size < 0:
        parser.error("repetition window must be positive and no-repeat ngram size cannot be negative")
    if args.temperature <= 0 or args.repetition_penalty <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature and repetition penalty must be positive; top-p must be in (0, 1]")

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

    if checkpoint_path.suffix.lower() == ".safetensors":
        from iva_nano.hf_compat import load_safetensors_checkpoint
        model_state, model_config, metadata = load_safetensors_checkpoint(
            checkpoint_path, device=device
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint["model"]
        model_config = checkpoint["model_config"]

    from iva_nano.hf_compat import audit_sentencepiece_tokenizer
    audit_sentencepiece_tokenizer(tokenizer_path, int(model_config["vocab_size"]))
    model = IVANano(model_config).to(device)
    model.load_state_dict(model_state, strict=True)
    if args.lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter, is_trainable=False)

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    generation_stats = {}
    answer = generate(
        model, tokenizer, args.prompt, device, args.max_new_tokens,
        args.temperature, args.top_k, args.top_p, args.repetition_penalty,
        args.repetition_window, args.no_repeat_ngram_size,
        generation_stats if args.report_stats else None,
    )
    print(answer)
    if args.report_stats:
        print(
            "IVA_GENERATION_STATS=" + json.dumps(generation_stats, ensure_ascii=False),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
