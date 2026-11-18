"""Safe Hugging Face ecosystem helpers for IVA Nano.

IVA's native tokenizer remains SentencePiece. These helpers validate that tokenizer
against the model vocabulary and load/export weights using the non-executable
Safetensors format.
"""
import hashlib
import json
from pathlib import Path

import sentencepiece as spm
import torch


def audit_sentencepiece_tokenizer(tokenizer_path, expected_vocab_size):
    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    processor = spm.SentencePieceProcessor(model_file=str(path))
    actual_vocab_size = int(processor.get_piece_size())
    if actual_vocab_size != int(expected_vocab_size):
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer has {actual_vocab_size} "
            f"pieces, model expects {expected_vocab_size}. Do not use this tokenizer "
            "with this checkpoint."
        )
    special_ids = {
        "unk_id": int(processor.unk_id()),
        "bos_id": int(processor.bos_id()),
        "eos_id": int(processor.eos_id()),
        "pad_id": int(processor.pad_id()),
    }
    for name, token_id in special_ids.items():
        if token_id >= actual_vocab_size or token_id < -1:
            raise ValueError(f"Invalid {name}: {token_id}")
    probe = "Проверка токенизатора IVA Nano."
    encoded = processor.encode(probe, out_type=int)
    if not encoded or any(token_id < 0 or token_id >= actual_vocab_size for token_id in encoded):
        raise ValueError("Tokenizer produced empty or out-of-range token IDs")
    decoded = processor.decode(encoded)
    if not decoded.strip():
        raise ValueError("Tokenizer failed the encode/decode smoke check")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "vocab_size": actual_vocab_size,
        "special_token_ids": special_ids,
        "probe_token_count": len(encoded),
        "probe_roundtrip_ok": True,
    }


def load_safetensors_checkpoint(path, device="cpu"):
    """Return (state_dict, model_config, metadata) from an IVA Safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = str(path)
    with safe_open(path, framework="pt", device=str(device)) as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != "iva-nano-safetensors-v1":
        raise ValueError("Unsupported or missing IVA Safetensors metadata")
    try:
        model_config = json.loads(metadata["model_config"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Safetensors file has no valid model_config metadata") from exc
    state = load_file(path, device=str(device))
    if model_config.get("tie_embeddings", True) and "lm_head.weight" not in state:
        if "token_embedding.weight" not in state:
            raise ValueError("Safetensors file is missing token_embedding.weight")
        state["lm_head.weight"] = state["token_embedding.weight"]
    return state, model_config, metadata


def export_safetensors_checkpoint(checkpoint, output_path):
    """Export model weights and architecture metadata, intentionally excluding optimizer state."""
    from safetensors.torch import save_file

    model_config = checkpoint.get("model_config")
    state = checkpoint.get("model")
    if not isinstance(model_config, dict) or not isinstance(state, dict):
        raise ValueError("Expected an IVA training checkpoint with model_config and model")
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    if model_config.get("tie_embeddings", True):
        embedding = tensors.get("token_embedding.weight")
        head = tensors.get("lm_head.weight")
        if embedding is not None and head is not None and torch.equal(embedding, head):
            tensors.pop("lm_head.weight")
    if not tensors:
        raise ValueError("Checkpoint contains no model tensors")
    metadata = {
        "format": "iva-nano-safetensors-v1",
        "model_config": json.dumps(model_config, ensure_ascii=False, sort_keys=True),
        "model_id": str(checkpoint.get("model_id", "iva-nano")),
        "step": str(int(checkpoint.get("step", 0))),
        "parameter_count": str(int(checkpoint.get("parameter_count", 0))),
        "tokenizer_path": str(checkpoint.get("tokenizer_path", "")),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata=metadata)
    return output
