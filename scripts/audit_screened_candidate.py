"""Aggregate quality audit for a screened candidate text file.

Streaming only. Emits aggregate metrics, never raw records, and never approves training.
"""
import argparse
import json
import re
from pathlib import Path

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
LONG_REPEAT = re.compile(r"(.)\1{11,}")


def audit(path: Path) -> dict:
    counts = {
        "records": 0,
        "blank_records": 0,
        "alphabetic_characters": 0,
        "cyrillic_characters": 0,
        "records_over_20000_chars": 0,
        "records_over_10000_chars": 0,
        "records_with_replacement_char": 0,
        "possible_email_records_remaining": 0,
        "possible_phone_records_remaining": 0,
        "long_character_repeat_records": 0,
        "max_record_characters": 0,
    }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = raw.rstrip("\r\n")
            counts["records"] += 1
            if not text.strip():
                counts["blank_records"] += 1
                continue
            length = len(text)
            counts["max_record_characters"] = max(counts["max_record_characters"], length)
            if length > 20_000:
                counts["records_over_20000_chars"] += 1
            if length > 10_000:
                counts["records_over_10000_chars"] += 1
            if "\ufffd" in text:
                counts["records_with_replacement_char"] += 1
            counts["alphabetic_characters"] += len(ALPHA.findall(text))
            counts["cyrillic_characters"] += len(CYRILLIC.findall(text))
            counts["possible_email_records_remaining"] += int(bool(EMAIL.search(text)))
            counts["possible_phone_records_remaining"] += int(bool(PHONE.search(text)))
            counts["long_character_repeat_records"] += int(bool(LONG_REPEAT.search(text)))

    letters = counts["alphabetic_characters"]
    counts["cyrillic_share_of_letters"] = (
        round(counts["cyrillic_characters"] / letters, 6) if letters else 0
    )
    return {
        "status": "candidate_quality_audit_only",
        "source_file": path.name,
        "metrics": counts,
        "training_approved": False,
        "limitations": [
            "Heuristics are not a complete privacy review.",
            "A high Cyrillic share does not establish factuality, quality, provenance, or rights.",
            "Line-based metrics depend on the source's record formatting.",
            "Human review remains required before any training approval.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Source file not found: {args.source}")
    result = audit(args.source)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
