"""Bounded sample audit for one Hugging Face dataset candidate.

Streams at most N rows, emits aggregate diagnostics only, and never changes
registry approval or training manifests. No raw sample text is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
GENERIC = (
    "конечно, помогу", "сейчас объясню", "вот прямой ответ",
    "обращайся", "добро пожаловать в чат", "sure, here is",
    "happy to help", "okay.",
)
ANSWER_KEYS = {"a", "answer", "answer_text", "assistant", "assistant_response", "completion", "completion_text", "full_answer", "output", "response", "response_text", "target", "verifiable_answer"}
DEFAULTS = {
    "d0rj_math_dataset_ru": ("default", "ru"),
    "s85io_math_logic_ru": ("default", "train"),
    "openai_gsm8k_math": ("main", "train"),
    "ttech_tmath_ru_eval": ("default", "train"),
}


def flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(v) for v in value)
    return "" if value is None else str(value)


def answer_text(row: dict[str, Any]) -> str:
    pieces = []
    for key, value in row.items():
        if key.lower() in ANSWER_KEYS:
            pieces.append(flatten(value))
        elif key.lower() in {"messages", "conversation", "conversations", "dialogue"} and isinstance(value, list):
            for message in value:
                if isinstance(message, dict):
                    role = str(message.get("role", message.get("from", message.get("speaker", "")))).lower()
                    if role in {"assistant", "bot", "gpt", "iva"}:
                        pieces.append(flatten(message.get("content", message.get("value", message.get("text", "")))))
        elif key.lower() == "dialog" and isinstance(value, list):
            roles = row.get("role", [])
            if isinstance(roles, list) and len(roles) == len(value):
                for role, utterance in zip(roles, value):
                    if str(role).lower() in {"assistant", "bot", "gpt", "iva"}:
                        pieces.append(flatten(utterance))
    return " ".join(piece for piece in pieces if piece).strip()


def audit(repo_id: str, config: str | None, split: str, max_rows: int) -> dict:
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    stream = load_dataset(repo_id, **kwargs)

    seen_rows: set[str] = set()
    seen_answers: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    field_types: dict[str, set[str]] = {}
    sampled = empty = chars = letters = cyrillic = duplicate_rows = answer_rows = 0
    email_rows = phone_rows = generic_answers = 0
    answer_lengths: list[int] = []

    for row in stream:
        if sampled >= max_rows:
            break
        if not isinstance(row, dict):
            row = {"text": row}
        sampled += 1
        if not row or not any(v not in (None, "", [], {}) for v in row.values()):
            empty += 1
        for key, value in row.items():
            field_types.setdefault(str(key), set()).add(type(value).__name__)
        lang = row.get("language", row.get("lang", row.get("locale", "")))
        if lang:
            language_counts[str(lang).lower()] += 1

        text = flatten(row)
        chars += len(text)
        alpha = [ch for ch in text if ch.isalpha()]
        letters += len(alpha)
        cyrillic += sum(bool(CYRILLIC.fullmatch(ch)) for ch in alpha)
        email_rows += int(bool(EMAIL.search(text)))
        phone_rows += int(bool(PHONE.search(text)))

        digest = hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()
        if digest in seen_rows:
            duplicate_rows += 1
        else:
            seen_rows.add(digest)

        answer = answer_text(row)
        if answer:
            normalized = " ".join(answer.casefold().split())
            seen_answers[normalized] += 1
            answer_lengths.append(len(answer))
            generic_answers += int(any(phrase in normalized for phrase in GENERIC))

    repeated_answers = sum(count - 1 for count in seen_answers.values() if count > 1)
    return {
        "dataset": repo_id,
        "config": config,
        "split": split,
        "sample_limit": max_rows,
        "sampled_rows": sampled,
        "fields": {key: sorted(values) for key, values in sorted(field_types.items())},
        "language_field_sample_counts": dict(language_counts),
        "cyrillic_letter_share": round(cyrillic / letters, 4) if letters else None,
        "average_record_characters": round(chars / sampled, 1) if sampled else 0,
        "answer_rows_detected": answer_rows,\n        "average_answer_characters": round(sum(answer_lengths) / len(answer_lengths), 1) if answer_lengths else None,
        "empty_rows": empty,
        "exact_duplicate_rows_in_sample": duplicate_rows,
        "repeated_answer_occurrences_in_sample": repeated_answers,
        "generic_answer_rows_in_sample": generic_answers,
        "possible_email_rows": email_rows,
        "possible_phone_rows": phone_rows,
        "raw_samples_saved": False,
        "approval_changed": False,
        "status": "sampled_pending_human_review",
        "limitations": [
            "This is a bounded sample, not a full-corpus audit.",
            "Duplicate and personal-data detectors are heuristic and may miss cases or flag false positives.",
            "No legal rights, factual correctness, mathematical proof validity, or safety guarantee is established.",
            "No row is approved or added to a training manifest.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="data/dataset_candidates.json")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--max-rows", type=int, default=250)
    parser.add_argument("--output", default="reports/dataset-candidate-audit.json")
    args = parser.parse_args()

    if not 1 <= args.max_rows <= 500:
        parser.error("--max-rows must be between 1 and 500")
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    entry = next((d for d in registry.get("datasets", []) if d.get("id") == args.dataset_id), None)
    if entry is None:
        parser.error(f"Unknown dataset id: {args.dataset_id}")
    if entry.get("status") not in {"candidate", "hold"}:
        parser.error("Only candidate/hold sources may be sampled")
    default_config, default_split = DEFAULTS.get(args.dataset_id, ("", "train"))
    config = args.config or default_config
    split = args.split or default_split

    try:
        report = audit(entry["hf_id"], config or None, split, args.max_rows)
    except Exception as exc:
        report = {
            "dataset": entry["hf_id"],
            "dataset_id": args.dataset_id,
            "config": config or None,
            "split": split,
            "status": "sample_audit_failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "approval_changed": False,
            "raw_samples_saved": False,
        }
        code = 1
    else:
        report["dataset_id"] = args.dataset_id
        report["registry_status"] = entry.get("status")
        report["declared_license"] = entry.get("declared_license")
        report["role"] = entry.get("role")
        code = 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
