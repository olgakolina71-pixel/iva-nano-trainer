"""Screen a quarantined candidate corpus for basic quality/privacy signals.

The report contains aggregate counts only. It never approves data or starts training.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
LONG_REPEAT = re.compile(r"(.)\1{11,}")


def audit(manifest_path, root_dir):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    root = Path(root_dir).resolve()
    results = []
    for entry in manifest.get("datasets", []):
        path = Path(entry["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError(f"Candidate path escapes quarantine directory: {path}")
        if not resolved.is_file():
            results.append({"id": entry.get("id"), "status": "missing_file"})
            continue

        seen = set()
        duplicate_scan_limit = 500000
        duplicate_scan_skipped = 0
        records = chars = letters = cyrillic = duplicates = 0
        emails = phones = low_cyrillic = repeated = 0

        def consume(text):
            nonlocal records, chars, letters, cyrillic, duplicates
            nonlocal emails, phones, low_cyrillic, repeated, duplicate_scan_skipped
            text = text.strip()
            if not text:
                return
            records += 1
            chars += len(text)
            alpha = [ch for ch in text if ch.isalpha()]
            letters += len(alpha)
            cyrillic_count = sum(bool(CYRILLIC.fullmatch(ch)) for ch in alpha)
            cyrillic += cyrillic_count
            if alpha and cyrillic_count / len(alpha) < 0.5:
                low_cyrillic += 1
            emails += int(bool(EMAIL.search(text)))
            phones += int(bool(PHONE.search(text)))
            repeated += int(bool(LONG_REPEAT.search(text)))
            digest = hashlib.sha256(text.casefold().encode("utf-8")).digest()
            if digest in seen:
                duplicates += 1
            elif len(seen) < duplicate_scan_limit:
                seen.add(digest)
            else:
                duplicate_scan_skipped += 1

        # Process line by line so very large plain-text corpora stay memory-safe.
        with resolved.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    consume(line.rstrip())

        results.append({
            "id": entry.get("id"), "use": entry.get("use"),
            "rights_status": entry.get("rights_status"),
            "records": records, "characters": chars,
            "cyrillic_letter_share": round(cyrillic / letters, 4) if letters else 0,
            "duplicate_records_detected": duplicates,
            "duplicate_scan_skipped_after_limit": duplicate_scan_skipped,
            "possible_email_records": emails, "possible_phone_records": phones,
            "low_cyrillic_records": low_cyrillic,
            "long_character_repeat_records": repeated,
            "review_required": True,
            "note": "Heuristic screening only; no automatic privacy, rights, or quality approval.",
        })
    return {"status": "quarantine_only", "sources": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/hf_manifest.json")
    parser.add_argument("--root", default="data/raw/hf")
    parser.add_argument("--output", default="reports/candidate-quality-audit.json")
    args = parser.parse_args()
    report = audit(args.manifest, args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
