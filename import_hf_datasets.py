"""Acquire candidate Hugging Face datasets into quarantine, never auto-approve them.

Install: pip install -r requirements.txt
Run: python import_hf_datasets.py
The script streams bounded rows, filters obvious non-Russian rows, deduplicates,
and writes per-source provenance. Every entry remains pending_rights_review because
a dataset-card license declaration is not independent proof of underlying rights.
"""
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/raw/hf")
MANIFEST = Path("data/hf_manifest.json")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
SPACE = re.compile(r"\s+")

SOURCES = [
    {
        "id": "ru_classic_cc0",
        "repo": "Imperius/ru-classic",
        "kind": "text",
        "split": "train",
        "configs": [None],
        "text_fields": ["text"],
        "max_records": 100000,
        "max_chars": 12000,
        "license": "CC0-1.0",
        "attribution": "Imperius/ru-classic; source and license as declared in its Hugging Face dataset card: https://huggingface.co/datasets/Imperius/ru-classic",
        "url": "https://huggingface.co/datasets/Imperius/ru-classic",
        "notes": "Russian classical literature; style-heavy, not a modern factual or conversational corpus. Dataset card declares underlying works public domain and compilation CC0; verify jurisdiction and provenance.",
    },
    {
        "id": "ru_edutexts_cc0",
        "repo": "nyuuzyou/edutexts",
        "kind": "educational",
        "split": "train",
        "configs": ["presentations", "documents"],
        "text_fields": ["slide_text", "document_text"],
        "max_records": 100000,
        "max_chars": 8000,
        "license": "CC0-1.0",
        "attribution": "nyuuzyou/edutexts; https://huggingface.co/datasets/nyuuzyou/edutexts",
        "url": "https://huggingface.co/datasets/nyuuzyou/edutexts",
        "notes": "Educational presentations and documents, primarily Russian with some Ukrainian and English. Dataset card declares CC0. Filtered by Cyrillic ratio; inspect quality and source provenance.",
    },
    {
        "id": "small_life_dialogues_cc0",
        "repo": "SonexaAI/Small-Life-Dataset-ru-eng",
        "kind": "jsonl",
        "split": "train",
        "configs": [None],
        "text_fields": ["question", "answer"],
        "max_records": 50000,
        "max_chars": 4000,
        "license": "CC0-1.0",
        "attribution": "SonexaAI, Small-Life-Dataset-ru-eng; https://huggingface.co/datasets/SonexaAI/Small-Life-Dataset-ru-eng",
        "url": "https://huggingface.co/datasets/SonexaAI/Small-Life-Dataset-ru-eng",
        "notes": "Synthetic short Russian/English prompt-response pairs. Dataset card declares CC0; repository file layout has changed before, so import may fail and must be reviewed.",
    },
]


def flatten(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(item) for item in value if item is not None)
    return ""


def russian_enough(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return sum(1 for ch in letters if CYRILLIC.fullmatch(ch)) / len(letters) >= 0.20


def clean(text, max_chars, min_chars=1):
    text = SPACE.sub(" ", text.replace("\x00", " ")).strip()
    if len(text) < min_chars:
        return ""
    return text[:max_chars]


def extract_rows(source):
    repo = source["repo"]
    if source["kind"] == "jsonl":
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/train.jsonl"
        stream = load_dataset("json", data_files=url, split="train", streaming=True)
        for row in stream:
            language = str(row.get("language", row.get("lang", ""))).lower()
            if language and language not in {"ru", "russian", "русский"}:
                continue
            q = clean(flatten(row.get("question", row.get("prompt", ""))), source["max_chars"], min_chars=1)
            a = clean(flatten(row.get("answer", row.get("response", ""))), source["max_chars"], min_chars=1)
            text = f"### Пользователь:\n{q}\n### IVA:\n{a}" if q and a else ""
            if text and russian_enough(text):
                yield text
        return

    for config in source["configs"]:
        kwargs = {"split": source["split"], "streaming": True}
        if config:
            kwargs["name"] = config
        stream = load_dataset(repo, **kwargs)
        for row in stream:
            parts = [clean(flatten(row.get(field, "")), source["max_chars"], min_chars=80) for field in source["text_fields"]]
            text = "\n".join(part for part in parts if part)
            if text and russian_enough(text):
                yield text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for source in SOURCES:
        target = OUT / f"{source['id']}.txt"
        seen = set()
        count = 0
        chars = 0
        try:
            with target.open("w", encoding="utf-8") as handle:
                for text in extract_rows(source):
                    digest_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if digest_key in seen:
                        continue
                    seen.add(digest_key)
                    handle.write(text + "\n\n")
                    count += 1
                    chars += len(text)
                    if count >= source["max_records"]:
                        break
        except Exception:
            target.unlink(missing_ok=True)
            raise
        if count == 0:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"{source['id']}: no usable Russian records; no manifest written")
        entries.append({
            "id": source["id"],
            "path": target.as_posix(),
            "license": source["license"],
            "attribution": source["attribution"],
            "source_url": source["url"],
            "use": "pending_rights_review",
            "rights_status": "dataset_card_claim_unverified",
            "records": count,
            "characters": chars,
            "notes": source["notes"],
        })
        print(f"{source['id']}: {count:,} records, {chars:,} characters")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"datasets": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
