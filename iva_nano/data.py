import json
import re
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


SFT_TURN = re.compile(
    r"### Пользователь:\s*\n(.*?)\n### IVA:\s*\n(.*?)(?=\n### Пользователь:\s*\n|$)",
    re.DOTALL,
)


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


def encode_training_texts(texts, tokenizer, assistant_only=False):
    """Encode pretraining text or assistant-only SFT targets.

    For SFT, user prompts remain in the input context but receive -100 labels,
    so cross-entropy trains only on IVA response tokens.
    """
    token_ids = []
    labels = []
    eos_id = tokenizer.eos_id()

    if not assistant_only:
        for text in texts:
            encoded = tokenizer.encode(text, out_type=int)
            token_ids.extend(encoded)
            token_ids.append(eos_id)
        return token_ids, token_ids.copy()

    examples = 0
    for source_text in texts:
        documents = re.split(r"\n\s*\n", source_text)
        for document in documents:
            for match in SFT_TURN.finditer(document.strip()):
                user_text = match.group(1).strip()
                assistant_text = match.group(2).strip()
                if not user_text or not assistant_text:
                    continue
                prompt = f"### Пользователь:\n{user_text}\n### IVA:\n"
                prompt_ids = tokenizer.encode(prompt, out_type=int)
                answer_ids = tokenizer.encode(assistant_text, out_type=int)
                if not prompt_ids or not answer_ids:
                    continue
                token_ids.extend(prompt_ids)
                labels.extend([-100] * len(prompt_ids))
                token_ids.extend(answer_ids)
                labels.extend(answer_ids)
                token_ids.append(eos_id)
                labels.append(eos_id)
                examples += 1

    if examples == 0:
        raise ValueError(
            "Assistant-only SFT requested, but no '### Пользователь:' / '### IVA:' "
            "conversation pairs were found. Prepare a reviewed dialogue corpus first."
        )
    return token_ids, labels


class TokenBlockDataset(Dataset):
    def __init__(self, texts, tokenizer_path: str, context_length: int, assistant_only=False):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        ids, labels = encode_training_texts(texts, self.sp, assistant_only=assistant_only)
        if len(ids) != len(labels):
            raise ValueError("Token and target-label lengths do not match")
        self.tokens = torch.tensor(ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.context_length = int(context_length)
        self.starts = []
        for start in range(0, len(self.tokens) - self.context_length, self.context_length):
            target = self.labels[start + 1:start + self.context_length + 1]
            if target.numel() == self.context_length and bool((target != -100).any()):
                self.starts.append(start)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        x = self.tokens[start:start + self.context_length]
        y = self.labels[start + 1:start + self.context_length + 1]
        return x, y
