import json
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


def load_approved_texts(data_dir: str, manifest_path: str):
    root = Path(data_dir)
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_file}. "
            "Create it and record license and attribution for every source."
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    entries = manifest.get("datasets", [])
    if not entries:
        raise ValueError("Manifest contains no datasets")

    texts = []
    for entry in entries:
        if not entry.get("license") or not entry.get("attribution"):
            raise ValueError(f"Dataset {entry.get('id', '<unknown>')} needs license and attribution")
        if entry.get("use") != "training":
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if root.resolve() not in path.resolve().parents:
            raise ValueError(f"Dataset file must be inside {root}: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            texts.append(text)
    if not texts:
        raise ValueError("No non-empty training text files were listed in the manifest")
    return texts


def train_tokenizer(texts, prefix: str, vocab_size: int, model_type: str = "bpe"):
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = prefix_path.with_suffix(".training.txt")
    corpus.write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix_path),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        hard_vocab_limit=False,
    )
    corpus.unlink(missing_ok=True)
    return str(prefix_path) + ".model"


class TokenBlockDataset(Dataset):
    def __init__(self, texts, tokenizer_path: str, context_length: int):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        ids = []
        for text in texts:
            ids.extend(self.sp.encode(text, out_type=int))
            ids.append(self.sp.eos_id())
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.context_length)

    def __getitem__(self, index):
        start = index * self.context_length
        chunk = self.tokens[start:start + self.context_length + 1]
        return chunk[:-1], chunk[1:]
