"""Create a privacy-safe index of quality-flagged candidate records.

Only line numbers and flag names are emitted; source text is never copied.
This is a review aid, not an automatic cleaner or training approval.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

LONG_REPEAT = re.compile(r"(.)\\1{11,}")


def inspect(source: Path, index_path: Path, report_path: Path) -> dict:
    totals = Counter()
    flagged_records = 0
    records = 0
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8", errors="replace") as src, index_path.open(
        "w", encoding="utf-8"
    ) as index:
        for line_number, raw in enumerate(src, start=1):
            records += 1
            value = raw.rstrip("\\r\\n")
            flags = []
            if len(value) > 20_000:
                flags.append("over_20000_characters")
            elif len(value) > 10_000:
                flags.append("over_10000_characters")
            if "\\ufffd" in value:
                flags.append("replacement_character")
            if LONG_REPEAT.search(value):
                flags.append("long_character_repeat")

            if flags:
                flagged_records += 1
                totals.update(flags)
                index.write(json.dumps(
                    {"line": line_number, "flags": flags},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\\n")

    report = {
        "status": "quality_flag_index_for_human_review",
        "source_file": source.name,
        "records_scanned": records,
        "flagged_records": flagged_records,
        "flag_counts": dict(sorted(totals.items())),
        "index_file": index_path.name,
        "raw_source_text_included": False,
        "training_approved": False,
        "limitations": [
            "Flags identify records for review; they do not prove a record is defective.",
            "No source text is copied into the index.",
            "Line numbers depend on the exact source snapshot and must not be reused with another version.",
            "Human review and rights approval remain mandatory.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Source file not found: {args.source}")
    result = inspect(args.source, args.index, args.report)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
