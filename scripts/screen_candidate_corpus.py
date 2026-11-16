"""Create a privacy-screened candidate copy from the quarantined Russian corpus.

This is a screening step only: it does not approve licensing or training use.
Original source files are never modified. Matching records are counted, not logged.
"""
import argparse
import json
import re
from pathlib import Path

EMAIL = re.compile(r"(?i)\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b")
PHONE = re.compile(r"(?<!\\w)(?:\\+?\\d[\\d ()-]{8,}\\d)(?!\\w)")


def screen(source: Path, output: Path) -> dict:
    counts = {
        "records_seen": 0,
        "kept_records": 0,
        "possible_email_records_quarantined": 0,
        "possible_phone_records_quarantined": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace") as src, output.open(
        "w", encoding="utf-8", newline="\\n"
    ) as dst:
        for line in src:
            record = line.strip()
            if not record:
                continue
            counts["records_seen"] += 1
            if EMAIL.search(record):
                counts["possible_email_records_quarantined"] += 1
                continue
            if PHONE.search(record):
                counts["possible_phone_records_quarantined"] += 1
                continue
            dst.write(record + "\\n")
            counts["kept_records"] += 1

    return {
        "status": "candidate_only_pending_human_review",
        "source_file": source.name,
        "output_file": output.name,
        "method": "Streaming record-level screening with the repository audit email/phone heuristics; source order preserved; duplicates retained.",
        "counts": counts,
        "output_bytes": output.stat().st_size,
        "rights_status": "dataset_card_claim_unverified",
        "training_approved": False,
        "limitations": [
            "Heuristic matches can be false positives; matched records are excluded only from this candidate copy.",
            "Duplicates, low-Cyrillic records and repeated-character records remain for separate review.",
            "No license, provenance, privacy, factuality or quality approval is implied.",
            "The source archive is not modified.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="cleaned/ru_classic_candidate.txt")
    parser.add_argument("--report", default="reports/ru_classic_cleaning_report.json")
    args = parser.parse_args()
    report = screen(Path(args.source), Path(args.output))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
