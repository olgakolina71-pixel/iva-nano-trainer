"""Normalize quarantined IVA text snapshots without approving their rights.

Supports newline-delimited text, JSON arrays/objects, and blank-line-separated
dialogue turns. Every source keeps its own provenance and existing rights gate.
"""
import argparse
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

SPACE = re.compile(r"\s+")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
EMAIL = re.compile(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b")
PHONE = re.compile(r"(?<!\\w)(?:\\+?7|8)?[\\s(.-]*\\d{3}[\\s).-]*\\d{3}[\\s.-]*\\d{2}[\\s.-]*\\d{2}(?!\\w)")
STUDENT_NAME_CLASS = re.compile(
    r"\\b[А-ЯЁ][а-яё-]{2,}\\s+[А-ЯЁ][а-яё-]{2,}\\s+\\d{1,2}\\s*[«\\\"]"
)
PERSONAL_LABEL = re.compile(
    r"\\b(?:ФИО|выполнил[а]?|подготовил[а]?|ученик|ученица)\\s*[:\\-]?\\s*"
    r"[А-ЯЁ][а-яё-]+(?:\\s+[А-ЯЁ][а-яё-]+){1,2}",
    re.IGNORECASE,
)


def has_personal_markers(text):
    return bool(
        EMAIL.search(text)
        or PHONE.search(text)
        or STUDENT_NAME_CLASS.search(text)
        or PERSONAL_LABEL.search(text)
    )


def flatten_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(flatten_strings(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten_strings(item))
        return result
    return []


def normalize_text(value):
    value = unicodedata.normalize("NFC", html.unescape(value))
    value = value.replace("\x00", " ")
    return SPACE.sub(" ", value).strip()


def source_documents(path, dataset_id):
    if dataset_id == "small_life_dialogues_cc0":
        parts = []
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    parts.append(line)
                elif parts:
                    yield "\n".join(parts)
                    parts = []
        if parts:
            yield "\n".join(parts)
        return

    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if dataset_id == "ru_edutexts_cc0":
                try:
                    decoded = json.loads(line)
                    parts = flatten_strings(decoded)
                    if parts:
                        yield "\n".join(parts)
                        continue
                except json.JSONDecodeError:
                    pass
            yield line


def prepare(input_dir, input_manifest, output_dir, output_manifest):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(input_manifest).read_text(encoding="utf-8"))
    entries = manifest.get("datasets", [])
    if not entries:
        raise ValueError("Input manifest contains no datasets")

    output_entries = []
    for entry in entries:
        dataset_id = entry.get("id")
        if not dataset_id or not entry.get("license") or not entry.get("attribution"):
            raise ValueError(f"Missing ID, license, or attribution: {entry}")
        source_path = input_dir / Path(entry["path"]).name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        target = output_dir / f"{dataset_id}.txt"
        seen = set()
        records = characters = duplicates = rejected = personal_rejected = 0
        with target.open("w", encoding="utf-8", newline="\n") as output:
            for raw in source_documents(source_path, dataset_id):
                text = normalize_text(raw)
                if len(text) < (1 if dataset_id == "small_life_dialogues_cc0" else 40):
                    rejected += 1
                    continue
                letters = [c for c in text if c.isalpha()]
                if letters and sum(bool(CYRILLIC.fullmatch(c)) for c in letters) / len(letters) < 0.20:
                    rejected += 1
                    continue
                digest = hashlib.sha256(text.casefold().encode("utf-8")).digest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                output.write(text + "\n\n")
                records += 1
                characters += len(text)

        if records == 0:
            target.unlink(missing_ok=True)
            raise ValueError(f"{dataset_id}: no usable records after normalization")

        output_entries.append({
            **entry,
            "path": (output_dir / target.name).as_posix(),
            "records": records,
            "characters": characters,
            "preparation": {
                "method": "json-aware-normalization-and-exact-dedup-v1",
                "duplicates_removed": duplicates,
                "records_rejected": rejected,\n                "records_rejected_personal_markers": personal_rejected,
            },
            # Deliberately preserve use/rights_status. This script never approves data.
        })
        print(
            f"{dataset_id}: {records:,} records, {characters:,} characters, "
            f"{duplicates:,} duplicates removed, {rejected:,} rejected, {personal_rejected:,} personal-data flagged; "
            f"rights={entry.get('rights_status', 'unspecified')}"
        )

    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps({"datasets": output_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(description="Normalize IVA candidate datasets without changing rights status.")
    parser.add_argument("--input-dir", default="data/raw/hf")
    parser.add_argument("--input-manifest", default="data/hf_manifest.json")
    parser.add_argument("--output-dir", default="data/normalized_candidates")
    parser.add_argument("--output-manifest", default="data/normalized_candidates/manifest.json")
    args = parser.parse_args()
    prepare(args.input_dir, args.input_manifest, args.output_dir, args.output_manifest)


if __name__ == "__main__":
    main()
